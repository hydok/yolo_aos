package hydok.yolo

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.util.Log
import androidx.camera.core.ImageProxy
import androidx.compose.ui.geometry.Offset
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.ByteArrayInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.zip.ZipInputStream
import kotlin.math.roundToInt

/** [box] is normalized (0..1) to the visible camera preview. */
class Detection(val box: RectF, val label: String, val score: Float)

/** [keypoints] are normalized to the preview; null when the joint is not visible. */
class Pose(val score: Float, val keypoints: List<Offset?>)

enum class YoloTask(val title: String, val model: String) {
    DETECT("검출", "yolo26n.tflite"),
    SEGMENT("분할", "yolo26n-seg.tflite"),
    DEPTH("깊이", "yolo26n-depth.tflite"),
    POSE("자세", "yolo26n-pose.tflite"),
}

/** Images cover exactly the visible camera preview. */
sealed interface YoloResult {
    class Detections(val detections: List<Detection>) : YoloResult
    class Segments(val detections: List<Detection>, val mask: Bitmap) : YoloResult
    class Depth(val image: Bitmap) : YoloResult
    class Poses(val poses: List<Pose>) : YoloResult
}

class YoloDetector(context: Context, private val task: YoloTask) : AutoCloseable {
    private val labels: Array<String>
    private val gpu: GpuDelegate?
    private val interpreter: Interpreter
    private val size: Int
    private val bitmap: Bitmap
    private val canvas: Canvas
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val pixels: IntArray
    private val input: ByteBuffer
    private val inputFloats: FloatBuffer
    private val outputShapes: List<IntArray>
    private val outputs: List<ByteBuffer>
    private val outputFloats: List<FloatBuffer>

    // Letterbox placement of the last frame inside the size x size model input.
    private var padX = 0f
    private var padY = 0f
    private var contentW = 0f
    private var contentH = 0f

    init {
        val bytes = context.assets.open(task.model).use { it.readBytes() }
        val names = readMetadata(bytes).getJSONObject("names")
        labels = Array(names.length()) { names.getString(it.toString()) }

        val model = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).put(bytes)
        var delegate: GpuDelegate? = null
        interpreter = try {
            delegate = GpuDelegate()
            Interpreter(model, Interpreter.Options().addDelegate(delegate))
        } catch (e: Exception) {
            Log.w(TAG, "GPU delegate unavailable, using CPU", e)
            delegate?.close()
            delegate = null
            Interpreter(model, Interpreter.Options().setNumThreads(4))
        }
        gpu = delegate
        Log.i(TAG, "${task.model} running on ${if (gpu != null) "GPU" else "CPU"}")

        // Input is [1, 3, size, size].
        size = interpreter.getInputTensor(0).shape()[2]
        bitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        canvas = Canvas(bitmap)
        pixels = IntArray(size * size)
        input = ByteBuffer.allocateDirect(3 * size * size * 4).order(ByteOrder.nativeOrder())
        inputFloats = input.asFloatBuffer()
        outputShapes = (0 until interpreter.outputTensorCount).map { interpreter.getOutputTensor(it).shape() }
        outputs = outputShapes.map {
            ByteBuffer.allocateDirect(it.fold(4) { acc, d -> acc * d }).order(ByteOrder.nativeOrder())
        }
        outputFloats = outputs.map { it.asFloatBuffer() }
    }

    // Ultralytics appends a zip holding metadata.json to the model file.
    private fun readMetadata(bytes: ByteArray): JSONObject {
        var i = bytes.size - 4
        while (i >= 0 && !(bytes[i] == 'P'.code.toByte() && bytes[i + 1] == 'K'.code.toByte() &&
                bytes[i + 2] == 3.toByte() && bytes[i + 3] == 4.toByte())
        ) i--
        require(i >= 0) { "${task.model} has no embedded metadata" }
        return ZipInputStream(ByteArrayInputStream(bytes, i, bytes.size - i)).use { zip ->
            zip.nextEntry
            JSONObject(zip.readBytes().decodeToString())
        }
    }

    fun run(image: ImageProxy): YoloResult {
        letterbox(image)
        interpreter.runForMultipleInputsOutputs(
            arrayOf(input),
            outputs.withIndex().associate { (i, buffer) -> i to buffer.rewind() }
        )
        return when (task) {
            YoloTask.DETECT -> YoloResult.Detections(boxes(0, labels.size).map { it.detection })
            YoloTask.SEGMENT -> segments()
            YoloTask.DEPTH -> YoloResult.Depth(depth())
            YoloTask.POSE -> YoloResult.Poses(poses())
        }
    }

    private fun letterbox(image: ImageProxy) {
        // Crop to the area visible in the preview, rotate upright, letterbox into size x size.
        val crop = image.cropRect
        val rotation = image.imageInfo.rotationDegrees.toFloat()
        val rotated = RectF(0f, 0f, crop.width().toFloat(), crop.height().toFloat())
        Matrix().apply { setRotate(rotation) }.mapRect(rotated)
        val scale = size / maxOf(rotated.width(), rotated.height())
        contentW = rotated.width() * scale
        contentH = rotated.height() * scale
        padX = (size - contentW) / 2
        padY = (size - contentH) / 2
        val matrix = Matrix().apply {
            setTranslate(-crop.left.toFloat(), -crop.top.toFloat())
            postRotate(rotation)
            postTranslate(-rotated.left, -rotated.top)
            postScale(scale, scale)
            postTranslate(padX, padY)
        }
        canvas.drawColor(Color.GRAY)
        canvas.save()
        canvas.clipRect(padX, padY, padX + contentW, padY + contentH)
        canvas.drawBitmap(image.toBitmap(), matrix, paint)
        canvas.restore()

        // Model input is NCHW float32 RGB in 0..1.
        bitmap.getPixels(pixels, 0, size, 0, 0, size, size)
        val area = size * size
        for (i in 0 until area) {
            val p = pixels[i]
            inputFloats.put(i, (p shr 16 and 0xFF) / 255f)
            inputFloats.put(area + i, (p shr 8 and 0xFF) / 255f)
            inputFloats.put(2 * area + i, (p and 0xFF) / 255f)
        }
    }

    // Undo the letterbox so coordinates are relative to the preview.
    private fun previewX(v: Float) = (v * size - padX) / contentW
    private fun previewY(v: Float) = (v * size - padY) / contentH

    /** A kept box plus the anchor it came from and its letterbox-space box. */
    private class Candidate(val anchor: Int, val raw: RectF, val detection: Detection)

    /**
     * Decodes output [index] laid out as [1, rows, anchors]: cx, cy, w, h (normalized to the
     * letterboxed input) then [classes] scores, followed by any task-specific rows. Applies per-class NMS.
     */
    private fun boxes(index: Int, classes: Int): List<Candidate> {
        val out = outputFloats[index]
        val anchors = outputShapes[index][2]
        fun at(row: Int, a: Int) = out.get(row * anchors + a)

        val candidates = ArrayList<Candidate>()
        for (a in 0 until anchors) {
            var best = 0
            var bestScore = 0f
            for (c in 0 until classes) {
                val s = at(4 + c, a)
                if (s > bestScore) {
                    bestScore = s
                    best = c
                }
            }
            if (bestScore < CONFIDENCE_THRESHOLD) continue
            val cx = at(0, a)
            val cy = at(1, a)
            val w = at(2, a)
            val h = at(3, a)
            val raw = RectF(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            val box = RectF(previewX(raw.left), previewY(raw.top), previewX(raw.right), previewY(raw.bottom))
            candidates += Candidate(a, raw, Detection(box, labels[best], bestScore))
        }

        candidates.sortByDescending { it.detection.score }
        val kept = ArrayList<Candidate>()
        for (c in candidates) {
            if (kept.none {
                    it.detection.label == c.detection.label && iou(it.detection.box, c.detection.box) > IOU_THRESHOLD
                }
            ) kept += c
        }
        return kept
    }

    // Outputs: [1, 4 + classes + 32, anchors] boxes with mask coefficients and [1, 32, mh, mw] prototypes.
    private fun segments(): YoloResult.Segments {
        val boxIndex = outputShapes.indexOfFirst { it.size == 3 }
        val protoIndex = outputShapes.indexOfFirst { it.size == 4 }
        val (_, protoCount, mh, mw) = outputShapes[protoIndex]
        val anchors = outputShapes[boxIndex][2]
        val out = outputFloats[boxIndex]
        val protos = outputFloats[protoIndex]
        val kept = boxes(boxIndex, labels.size)

        val maskPixels = IntArray(mw * mh)
        val coefficients = FloatArray(protoCount)
        kept.forEachIndexed { n, c ->
            for (k in 0 until protoCount) coefficients[k] = out.get((4 + labels.size + k) * anchors + c.anchor)
            val color = MASK_COLORS[n % MASK_COLORS.size]
            // Each mask is the coefficient-weighted sum of prototypes, kept only inside its box.
            val x0 = (c.raw.left * mw).toInt().coerceIn(0, mw - 1)
            val x1 = (c.raw.right * mw).toInt().coerceIn(0, mw - 1)
            val y0 = (c.raw.top * mh).toInt().coerceIn(0, mh - 1)
            val y1 = (c.raw.bottom * mh).toInt().coerceIn(0, mh - 1)
            for (y in y0..y1) for (x in x0..x1) {
                var v = 0f
                for (k in 0 until protoCount) v += coefficients[k] * protos.get(k * mh * mw + y * mw + x)
                if (v > 0f) maskPixels[y * mw + x] = color
            }
        }
        val mask = Bitmap.createBitmap(maskPixels, mw, mh, Bitmap.Config.ARGB_8888)
        return YoloResult.Segments(kept.map { it.detection }, cropContent(mask))
    }

    // Output: [1, 1, size, size] depth in meters (larger is farther), colored near = red, far = blue.
    private fun depth(): Bitmap {
        val out = outputFloats[0]
        // Skip a thin margin: the model reads the gray letterbox edge as very near.
        val x0 = padX.roundToInt() + if (padX > 0f) EDGE else 0
        val y0 = padY.roundToInt() + if (padY > 0f) EDGE else 0
        val w = (contentW.roundToInt() - if (padX > 0f) 2 * EDGE else 0).coerceAtMost(size - x0)
        val h = (contentH.roundToInt() - if (padY > 0f) 2 * EDGE else 0).coerceAtMost(size - y0)
        var min = Float.MAX_VALUE
        var max = 0f
        for (y in y0 until y0 + h) for (x in x0 until x0 + w) {
            val d = out.get(y * size + x)
            if (d < min) min = d
            if (d > max) max = d
        }
        val range = (max - min).coerceAtLeast(1e-6f)
        val colors = IntArray(w * h)
        for (y in 0 until h) for (x in 0 until w) {
            val t = (out.get((y0 + y) * size + x0 + x) - min) / range
            colors[y * w + x] = DEPTH_COLORS[(t * 255).toInt().coerceIn(0, 255)]
        }
        return Bitmap.createBitmap(colors, w, h, Bitmap.Config.ARGB_8888)
    }

    // Output: [1, 4 + 1 + 17 * 3, anchors]: box, person score, then x, y, visibility per keypoint.
    private fun poses(): List<Pose> {
        val out = outputFloats[0]
        val anchors = outputShapes[0][2]
        val keypointCount = (outputShapes[0][1] - 5) / 3
        return boxes(0, 1).map { c ->
            Pose(c.detection.score, List(keypointCount) { k ->
                val row = 5 + k * 3
                if (out.get((row + 2) * anchors + c.anchor) < KEYPOINT_THRESHOLD) null
                else Offset(
                    previewX(out.get(row * anchors + c.anchor)),
                    previewY(out.get((row + 1) * anchors + c.anchor))
                )
            })
        }
    }

    /** Crops a bitmap covering the whole letterboxed input to the part that shows the camera frame. */
    private fun cropContent(full: Bitmap): Bitmap {
        val sx = full.width.toFloat() / size
        val sy = full.height.toFloat() / size
        val x = (padX * sx).roundToInt()
        val y = (padY * sy).roundToInt()
        val w = (contentW * sx).roundToInt().coerceIn(1, full.width - x)
        val h = (contentH * sy).roundToInt().coerceIn(1, full.height - y)
        return Bitmap.createBitmap(full, x, y, w, h)
    }

    override fun close() {
        interpreter.close()
        gpu?.close()
    }

    private companion object {
        const val TAG = "YoloDetector"
        const val CONFIDENCE_THRESHOLD = 0.4f
        const val IOU_THRESHOLD = 0.5f
        const val KEYPOINT_THRESHOLD = 0.5f
        const val EDGE = 4

        val MASK_COLORS = intArrayOf(
            0x9900E676.toInt(), 0x99FF4081.toInt(), 0x99448AFF.toInt(),
            0x99FFD740.toInt(), 0x99E040FB.toInt(), 0x9900E5FF.toInt(),
        )

        // Hue 0 (red, near) to 240 (blue, far).
        val DEPTH_COLORS = IntArray(256) { Color.HSVToColor(floatArrayOf(it / 255f * 240f, 1f, 1f)) }
    }
}

fun iou(a: RectF, b: RectF): Float {
    val w = minOf(a.right, b.right) - maxOf(a.left, b.left)
    val h = minOf(a.bottom, b.bottom) - maxOf(a.top, b.top)
    if (w <= 0f || h <= 0f) return 0f
    val inter = w * h
    return inter / (a.width() * a.height() + b.width() * b.height() - inter)
}
