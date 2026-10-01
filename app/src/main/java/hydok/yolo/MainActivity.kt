package hydok.yolo

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.ImageAnalysis
import androidx.camera.view.LifecycleCameraController
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedCard
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.runtime.withFrameNanos
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import hydok.yolo.mediapipe.MediaPipeScreen
import hydok.yolo.ui.theme.YoloCameraTheme
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            YoloCameraTheme {
                CameraScreen()
            }
        }
    }
}

@Composable
fun CameraScreen() {
    val context = LocalContext.current
    var hasPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED
        )
    }
    val launcher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { hasPermission = it }

    LaunchedEffect(Unit) {
        if (!hasPermission) launcher.launch(Manifest.permission.CAMERA)
    }

    var engine by rememberSaveable { mutableStateOf<Engine?>(null) }
    BackHandler(enabled = engine != null) { engine = null }

    if (!hasPermission) {
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Text(text = "카메라 권한이 필요합니다")
        }
        return
    }
    when (engine) {
        null -> HomeScreen(onSelect = { engine = it })
        Engine.YOLO -> CameraPreview(modifier = Modifier.fillMaxSize())
        Engine.MEDIAPIPE -> MediaPipeScreen(modifier = Modifier.fillMaxSize())
    }
}

enum class Engine { YOLO, MEDIAPIPE }

@Composable
fun HomeScreen(onSelect: (Engine) -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .systemBarsPadding()
            .padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp, Alignment.CenterVertically)
    ) {
        Text(text = "사용할 엔진을 선택하세요", style = MaterialTheme.typography.headlineSmall)
        EngineCard(
            title = "YOLO26n",
            description = "검출 · 분할 · 깊이 · 자세",
            onClick = { onSelect(Engine.YOLO) }
        )
        EngineCard(
            title = "MediaPipe",
            description = "배경 분리 · 사물 검출 · 이미지 분류 · 얼굴 · 손·제스처 · 자세",
            onClick = { onSelect(Engine.MEDIAPIPE) }
        )
    }
}

@Composable
private fun EngineCard(title: String, description: String, onClick: () -> Unit) {
    OutlinedCard(onClick = onClick, modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(20.dp)) {
            Text(text = title, style = MaterialTheme.typography.titleLarge)
            Text(text = description, style = MaterialTheme.typography.bodyMedium)
        }
    }
}

@Composable
fun CameraPreview(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val controller = remember { LifecycleCameraController(context) }
    var task by rememberSaveable { mutableStateOf(YoloTask.DETECT) }
    var result by remember { mutableStateOf<YoloResult?>(null) }

    DisposableEffect(Unit) {
        controller.imageAnalysisOutputImageFormat = ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888
        controller.bindToLifecycle(lifecycleOwner)
        onDispose { controller.unbind() }
    }
    DisposableEffect(task) {
        result = null
        val executor = Executors.newSingleThreadExecutor()
        val detector = lazy { YoloDetector(context, task) }
        controller.setImageAnalysisAnalyzer(executor) { image ->
            image.use { result = detector.value.run(it) }
        }
        onDispose {
            controller.clearImageAnalysisAnalyzer()
            executor.execute { if (detector.isInitialized()) detector.value.close() }
            executor.shutdown()
        }
    }

    Box(modifier = modifier) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { PreviewView(it).apply { this.controller = controller } }
        )
        when (val r = result) {
            is YoloResult.Detections -> DetectionOverlay(detections = r.detections, modifier = Modifier.fillMaxSize())
            is YoloResult.Segments -> {
                ImageOverlay(image = r.mask, modifier = Modifier.fillMaxSize())
                DetectionOverlay(detections = r.detections, modifier = Modifier.fillMaxSize())
            }
            is YoloResult.Depth -> ImageOverlay(image = r.image, modifier = Modifier.fillMaxSize())
            is YoloResult.Poses -> PoseOverlay(poses = r.poses, modifier = Modifier.fillMaxSize())
            null -> Unit
        }
        Row(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.85f))
                .navigationBarsPadding()
                .padding(horizontal = 12.dp, vertical = 8.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            YoloTask.entries.forEach { t ->
                FilterChip(selected = t == task, onClick = { task = t }, label = { Text(t.title) })
            }
        }
    }
}

/** Draws an image that covers the visible preview, stretched to fill it. */
@Composable
private fun ImageOverlay(image: Bitmap, modifier: Modifier = Modifier) {
    Canvas(modifier = modifier) {
        drawImage(
            image = image.asImageBitmap(),
            dstSize = IntSize(size.width.toInt(), size.height.toInt())
        )
    }
}

// COCO keypoint pairs: face, arms, torso, legs.
private val SKELETON = listOf(
    0 to 1, 0 to 2, 1 to 3, 2 to 4, 3 to 5, 4 to 6,
    5 to 6, 5 to 7, 7 to 9, 6 to 8, 8 to 10,
    5 to 11, 6 to 12, 11 to 12,
    11 to 13, 13 to 15, 12 to 14, 14 to 16,
)

@Composable
private fun PoseOverlay(poses: List<Pose>, modifier: Modifier = Modifier) {
    Canvas(modifier = modifier) {
        fun point(o: Offset) = Offset(o.x * size.width, o.y * size.height)
        poses.forEach { pose ->
            SKELETON.forEach { (a, b) ->
                val p = pose.keypoints.getOrNull(a) ?: return@forEach
                val q = pose.keypoints.getOrNull(b) ?: return@forEach
                drawLine(Color.Green, point(p), point(q), strokeWidth = 3.dp.toPx())
            }
            pose.keypoints.filterNotNull().forEach {
                drawCircle(Color.Red, radius = 4.dp.toPx(), center = point(it))
            }
        }
    }
}

private const val SMOOTHING = 0.3f

/** [shown] is the box currently drawn; it eases toward [target] every frame. */
private class Track(val shown: RectF, val target: Detection)

@Composable
fun DetectionOverlay(detections: List<Detection>, modifier: Modifier = Modifier) {
    val textMeasurer = rememberTextMeasurer()
    val labelStyle = TextStyle(color = Color.Black, background = Color.Green)

    // Boxes on screen glide toward the latest detections instead of jumping once per inference.
    val tracks = remember { mutableListOf<Track>() }
    var frame by remember { mutableLongStateOf(0L) }
    LaunchedEffect(detections) {
        val previous = tracks.toMutableList()
        tracks.clear()
        detections.forEach { d ->
            val match = previous
                .filter { it.target.label == d.label && iou(it.target.box, d.box) > 0.2f }
                .maxByOrNull { iou(it.target.box, d.box) }
            previous.remove(match)
            tracks += Track(match?.shown ?: RectF(d.box), d)
        }
    }
    LaunchedEffect(Unit) {
        while (true) withFrameNanos { frame = it }
    }

    Canvas(modifier = modifier) {
        frame // redraw every display frame
        tracks.forEach { t ->
            val shown = t.shown
            val box = t.target.box
            shown.left += (box.left - shown.left) * SMOOTHING
            shown.top += (box.top - shown.top) * SMOOTHING
            shown.right += (box.right - shown.right) * SMOOTHING
            shown.bottom += (box.bottom - shown.bottom) * SMOOTHING
            val d = Detection(shown, t.target.label, t.target.score)
            val topLeft = Offset(d.box.left * size.width, d.box.top * size.height)
            drawRect(
                color = Color.Green,
                topLeft = topLeft,
                size = Size(d.box.width() * size.width, d.box.height() * size.height),
                style = Stroke(width = 2.dp.toPx())
            )
            drawText(
                textLayoutResult = textMeasurer.measure(
                    "${d.label} ${(d.score * 100).toInt()}%",
                    labelStyle
                ),
                topLeft = topLeft
            )
        }
    }
}
