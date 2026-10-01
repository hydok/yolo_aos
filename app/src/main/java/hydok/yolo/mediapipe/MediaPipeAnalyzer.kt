package hydok.yolo.mediapipe

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.RectF
import androidx.camera.core.ImageProxy
import androidx.compose.ui.geometry.Offset
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.ByteBufferExtractor
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.gesturerecognizer.GestureRecognizer
import com.google.mediapipe.tasks.vision.imageclassifier.ImageClassifier
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenter
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import hydok.yolo.Detection
import java.nio.ByteOrder
import java.nio.FloatBuffer

enum class MpMode(val title: String, val model: String) {
    SEGMENT("배경 분리", "selfie_segmenter.tflite"),
    OBJECT("사물 검출", "efficientdet_lite0.tflite"),
    CLASSIFY("이미지 분류", "efficientnet_lite0.tflite"),
    FACE("얼굴", "face_landmarker.task"),
    HAND("손·제스처", "gesture_recognizer.task"),
    POSE("자세", "pose_landmarker_lite.task"),
}

/** All coordinates are normalized (0..1) to the visible camera preview, not mirrored. */
sealed interface MpResult {
    class Segmentation(val image: Bitmap) : MpResult
    class Objects(val detections: List<Detection>) : MpResult
    class Classes(val labels: List<Pair<String, Float>>) : MpResult
    class Faces(val faces: List<List<Offset>>, val expressions: List<Pair<String, Float>>) : MpResult
    class Hands(val hands: List<List<Offset>>, val gestures: List<Pair<String, Float>>) : MpResult
    class Poses(val poses: List<List<Offset>>) : MpResult
}

class MediaPipeAnalyzer(
    context: Context,
    mode: MpMode,
    private val blurBackground: () -> Boolean,
) : AutoCloseable {
    private val base = BaseOptions.builder().setModelAssetPath(mode.model).build()
    private val video = RunningMode.VIDEO

    private val task: AutoCloseable = when (mode) {
        MpMode.SEGMENT -> ImageSegmenter.createFromOptions(
            context,
            ImageSegmenter.ImageSegmenterOptions.builder().setBaseOptions(base).setRunningMode(video)
                .setOutputConfidenceMasks(true).setOutputCategoryMask(false).build()
        )
        MpMode.OBJECT -> ObjectDetector.createFromOptions(
            context,
            ObjectDetector.ObjectDetectorOptions.builder().setBaseOptions(base).setRunningMode(video)
                .setScoreThreshold(0.4f).setMaxResults(10).build()
        )
        MpMode.CLASSIFY -> ImageClassifier.createFromOptions(
            context,
            ImageClassifier.ImageClassifierOptions.builder().setBaseOptions(base).setRunningMode(video)
                .setMaxResults(3).build()
        )
        MpMode.FACE -> FaceLandmarker.createFromOptions(
            context,
            FaceLandmarker.FaceLandmarkerOptions.builder().setBaseOptions(base).setRunningMode(video)
                .setNumFaces(2).setOutputFaceBlendshapes(true).build()
        )
        MpMode.HAND -> GestureRecognizer.createFromOptions(
            context,
            GestureRecognizer.GestureRecognizerOptions.builder().setBaseOptions(base).setRunningMode(video)
                .setNumHands(2).build()
        )
        MpMode.POSE -> PoseLandmarker.createFromOptions(
            context,
            PoseLandmarker.PoseLandmarkerOptions.builder().setBaseOptions(base).setRunningMode(video)
                .build()
        )
    }

    fun analyze(image: ImageProxy): MpResult {
        val frame = image.toUprightBitmap()
        val input = BitmapImageBuilder(frame).build()
        val timestampMs = image.imageInfo.timestamp / 1_000_000
        return when (val t = task) {
            is ImageSegmenter -> {
                val mask = t.segmentForVideo(input, timestampMs).confidenceMasks().get().last()
                val buffer = ByteBufferExtractor.extract(mask).order(ByteOrder.nativeOrder()).asFloatBuffer()
                MpResult.Segmentation(composite(frame, buffer, mask.width, mask.height))
            }
            is ObjectDetector -> MpResult.Objects(
                t.detectForVideo(input, timestampMs).detections().map { d ->
                    val b = d.boundingBox()
                    val top = d.categories().first()
                    Detection(
                        RectF(b.left / frame.width, b.top / frame.height, b.right / frame.width, b.bottom / frame.height),
                        top.categoryName(),
                        top.score()
                    )
                }
            )
            is ImageClassifier -> MpResult.Classes(
                t.classifyForVideo(input, timestampMs).classificationResult().classifications()
                    .first().categories().map { it.categoryName() to it.score() }
            )
            is FaceLandmarker -> {
                val r = t.detectForVideo(input, timestampMs)
                val expressions = r.faceBlendshapes().orElse(emptyList()).firstOrNull().orEmpty()
                    .filter { it.categoryName() != "_neutral" }
                    .sortedByDescending { it.score() }
                    .take(3)
                    .map { it.categoryName() to it.score() }
                MpResult.Faces(r.faceLandmarks().map { it.toOffsets() }, expressions)
            }
            is GestureRecognizer -> {
                val r = t.recognizeForVideo(input, timestampMs)
                MpResult.Hands(
                    r.landmarks().map { it.toOffsets() },
                    r.gestures().map { it.first().categoryName() to it.first().score() }
                )
            }
            is PoseLandmarker -> MpResult.Poses(
                t.detectForVideo(input, timestampMs).landmarks().map { it.toOffsets() }
            )
            else -> error("Unknown task $t")
        }
    }

    // Keeps the person from the frame and replaces the background with black or a blurred copy.
    private fun composite(frame: Bitmap, mask: FloatBuffer, width: Int, height: Int): Bitmap {
        val src = if (frame.width == width && frame.height == height) frame
        else Bitmap.createScaledBitmap(frame, width, height, true)
        val pixels = IntArray(width * height)
        src.getPixels(pixels, 0, width, 0, 0, width, height)
        val background = if (blurBackground()) {
            val small = Bitmap.createScaledBitmap(src, maxOf(1, width / 16), maxOf(1, height / 16), true)
            IntArray(width * height).also {
                Bitmap.createScaledBitmap(small, width, height, true).getPixels(it, 0, width, 0, 0, width, height)
            }
        } else null

        for (i in pixels.indices) {
            val a = mask.get(i).coerceIn(0f, 1f)
            val p = pixels[i]
            val b = background?.get(i) ?: 0
            val r = ((p shr 16 and 0xFF) * a + (b shr 16 and 0xFF) * (1 - a)).toInt()
            val g = ((p shr 8 and 0xFF) * a + (b shr 8 and 0xFF) * (1 - a)).toInt()
            val bl = ((p and 0xFF) * a + (b and 0xFF) * (1 - a)).toInt()
            pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or bl
        }
        return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
    }

    override fun close() = task.close()
}

/** Crops to the area visible in the preview and rotates it upright. */
private fun ImageProxy.toUprightBitmap(): Bitmap {
    val crop = cropRect
    val matrix = Matrix().apply { postRotate(imageInfo.rotationDegrees.toFloat()) }
    return Bitmap.createBitmap(toBitmap(), crop.left, crop.top, crop.width(), crop.height(), matrix, true)
}

private fun List<NormalizedLandmark>.toOffsets() = map { Offset(it.x(), it.y()) }
