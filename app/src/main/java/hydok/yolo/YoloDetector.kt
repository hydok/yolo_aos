package hydok.yolo

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import androidx.camera.core.ImageProxy
import android.util.Log
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.ByteArrayInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.zip.ZipInputStream

/** [box] is normalized (0..1) to the visible camera preview. */
class Detection(val box: RectF, val label: String, val score: Float)

class YoloDetector(context: Context) : AutoCloseable {
    private val labels: Array<String>
    private val gpu: GpuDelegate?
    private val interpreter: Interpreter
    private val size: Int
    private val anchors: Int
    private val bitmap: Bitmap
    private val canvas: Canvas
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val pixels: IntArray
    private val input: ByteBuffer
    private val inputFloats: FloatBuffer
    private val output: Array<Array<FloatArray>>

    init {
        val bytes = context.assets.open(MODEL).use { it.readBytes() }
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
        Log.i(TAG, "Running on ${if (gpu != null) "GPU" else "CPU"}")

        // Input is [1, 3, size, size], output is [1, 4 + classes, anchors].
        size = interpreter.getInputTensor(0).shape()[2]
        anchors = interpreter.getOutputTensor(0).shape()[2]
        bitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        canvas = Canvas(bitmap)
        pixels = IntArray(size * size)
        input = ByteBuffer.allocateDirect(3 * size * size * 4).order(ByteOrder.nativeOrder())
        inputFloats = input.asFloatBuffer()
        output = Array(1) { Array(4 + labels.size) { FloatArray(anchors) } }
    }

    // Ultralytics appends a zip holding metadata.json to the model file.
    private fun readMetadata(bytes: ByteArray): JSONObject {
        var i = bytes.size - 4
        while (i >= 0 && !(bytes[i] == 'P'.code.toByte() && bytes[i + 1] == 'K'.code.toByte() &&
                bytes[i + 2] == 3.toByte() && bytes[i + 3] == 4.toByte())
        ) i--
        require(i >= 0) { "$MODEL has no embedded metadata" }
        return ZipInputStream(ByteArrayInputStream(bytes, i, bytes.size - i)).use { zip ->
            zip.nextEntry
            JSONObject(zip.readBytes().decodeToString())
        }
    }

    fun detect(image: ImageProxy): List<Detection> {
        // Crop to the area visible in the preview, rotate upright, letterbox into size x size.
        val crop = image.cropRect
        val rotation = image.imageInfo.rotationDegrees.toFloat()
        val rotated = RectF(0f, 0f, crop.width().toFloat(), crop.height().toFloat())
        Matrix().apply { setRotate(rotation) }.mapRect(rotated)
        val scale = size / maxOf(rotated.width(), rotated.height())
        val contentW = rotated.width() * scale
        val contentH = rotated.height() * scale
        val padX = (size - contentW) / 2
        val padY = (size - contentH) / 2
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
        interpreter.run(input, output)

        // Per anchor: cx, cy, w, h (normalized) then one score per class.
        val out = output[0]
        val candidates = ArrayList<Detection>()
        for (a in 0 until anchors) {
            var best = 0
            var bestScore = 0f
            for (c in labels.indices) {
                val s = out[4 + c][a]
                if (s > bestScore) {
                    bestScore = s
                    best = c
                }
            }
            if (bestScore < CONFIDENCE_THRESHOLD) continue
            // Undo the letterbox so the box is relative to the preview.
            val cx = (out[0][a] * size - padX) / contentW
            val cy = (out[1][a] * size - padY) / contentH
            val w = out[2][a] * size / contentW
            val h = out[3][a] * size / contentH
            candidates += Detection(
                RectF(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
                labels[best],
                bestScore
            )
        }

        // Per-class NMS.
        candidates.sortByDescending { it.score }
        val kept = ArrayList<Detection>()
        for (d in candidates) {
            if (kept.none { it.label == d.label && iou(it.box, d.box) > IOU_THRESHOLD }) kept += d
        }
        return kept
    }

    override fun close() {
        interpreter.close()
        gpu?.close()
    }

    private companion object {
        const val TAG = "YoloDetector"
        const val MODEL = "yolo11n.tflite"
        const val CONFIDENCE_THRESHOLD = 0.4f
        const val IOU_THRESHOLD = 0.5f
    }
}

fun iou(a: RectF, b: RectF): Float {
    val w = minOf(a.right, b.right) - maxOf(a.left, b.left)
    val h = minOf(a.bottom, b.bottom) - maxOf(a.top, b.top)
    if (w <= 0f || h <= 0f) return 0f
    val inter = w * h
    return inter / (a.width() * a.height() + b.width() * b.height() - inter)
}
