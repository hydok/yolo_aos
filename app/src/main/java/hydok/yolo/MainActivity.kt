package hydok.yolo

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.RectF
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.ImageAnalysis
import androidx.camera.view.LifecycleCameraController
import androidx.camera.view.PreviewView
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedCard
import androidx.compose.material3.Text
import androidx.compose.runtime.saveable.rememberSaveable
import hydok.yolo.mediapipe.MediaPipeScreen
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.withFrameNanos
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
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
            title = "YOLO11n",
            description = "사물 검출 (COCO 80종)",
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
    var detections by remember { mutableStateOf(emptyList<Detection>()) }

    DisposableEffect(Unit) {
        val executor = Executors.newSingleThreadExecutor()
        val detector = lazy { YoloDetector(context) }
        controller.imageAnalysisOutputImageFormat = ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888
        controller.setImageAnalysisAnalyzer(executor) { image ->
            image.use { detections = detector.value.detect(it) }
        }
        controller.bindToLifecycle(lifecycleOwner)
        onDispose {
            controller.clearImageAnalysisAnalyzer()
            controller.unbind()
            executor.execute { if (detector.isInitialized()) detector.value.close() }
            executor.shutdown()
        }
    }

    Box(modifier = modifier) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { PreviewView(it).apply { this.controller = controller } }
        )
        DetectionOverlay(detections = detections, modifier = Modifier.fillMaxSize())
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
