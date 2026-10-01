package hydok.yolo.mediapipe

import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.view.LifecycleCameraController
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.scale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import java.util.concurrent.Executors

@Composable
fun MediaPipeScreen(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val controller = remember { LifecycleCameraController(context) }
    var mode by rememberSaveable { mutableStateOf(MpMode.SEGMENT) }
    var frontCamera by rememberSaveable { mutableStateOf(false) }
    val blur = remember { mutableStateOf(false) }
    var result by remember { mutableStateOf<MpResult?>(null) }

    DisposableEffect(Unit) {
        controller.imageAnalysisOutputImageFormat = ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888
        controller.bindToLifecycle(lifecycleOwner)
        onDispose { controller.unbind() }
    }
    LaunchedEffect(frontCamera) {
        controller.cameraSelector =
            if (frontCamera) CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA
    }
    DisposableEffect(mode) {
        result = null
        val executor = Executors.newSingleThreadExecutor()
        val analyzer = lazy { MediaPipeAnalyzer(context, mode) { blur.value } }
        controller.setImageAnalysisAnalyzer(executor) { image ->
            image.use { result = analyzer.value.analyze(it) }
        }
        onDispose {
            controller.clearImageAnalysisAnalyzer()
            executor.execute { if (analyzer.isInitialized()) analyzer.value.close() }
            executor.shutdown()
        }
    }

    Box(modifier = modifier) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { PreviewView(it).apply { this.controller = controller } }
        )
        MpOverlay(result = result, mirror = frontCamera, modifier = Modifier.fillMaxSize())
        MpLabels(
            result = result,
            modifier = Modifier.align(Alignment.TopStart).statusBarsPadding().padding(12.dp)
        )
        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.85f))
                .navigationBarsPadding()
                .padding(horizontal = 12.dp, vertical = 8.dp)
        ) {
            Row(
                modifier = Modifier.horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                MpMode.entries.forEach { m ->
                    FilterChip(selected = m == mode, onClick = { mode = m }, label = { Text(m.title) })
                }
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                if (mode == MpMode.SEGMENT) {
                    FilterChip(
                        selected = blur.value,
                        onClick = { blur.value = !blur.value },
                        label = { Text("배경 흐림") }
                    )
                }
                TextButton(onClick = { frontCamera = !frontCamera }) {
                    Text(if (frontCamera) "후면 카메라로" else "전면 카메라로")
                }
            }
        }
    }
}

@Composable
private fun MpOverlay(result: MpResult?, mirror: Boolean, modifier: Modifier = Modifier) {
    val textMeasurer = rememberTextMeasurer()
    val labelStyle = TextStyle(color = Color.Black, background = Color.Green)
    Canvas(modifier = modifier) {
        // The front camera preview is mirrored; analysis frames are not.
        fun x(v: Float) = (if (mirror) 1 - v else v) * size.width
        fun point(o: Offset) = Offset(x(o.x), o.y * size.height)

        when (result) {
            is MpResult.Segmentation -> scale(scaleX = if (mirror) -1f else 1f, scaleY = 1f) {
                drawImage(
                    image = result.image.asImageBitmap(),
                    dstSize = IntSize(size.width.toInt(), size.height.toInt())
                )
            }
            is MpResult.Objects -> result.detections.forEach { d ->
                val topLeft = Offset(minOf(x(d.box.left), x(d.box.right)), d.box.top * size.height)
                drawRect(
                    color = Color.Green,
                    topLeft = topLeft,
                    size = Size(d.box.width() * size.width, d.box.height() * size.height),
                    style = Stroke(width = 2.dp.toPx())
                )
                drawText(
                    textLayoutResult = textMeasurer.measure("${d.label} ${(d.score * 100).toInt()}%", labelStyle),
                    topLeft = topLeft
                )
            }
            is MpResult.Faces -> result.faces.forEach { face ->
                face.forEach { drawCircle(Color.Cyan, radius = 1.dp.toPx(), center = point(it)) }
            }
            is MpResult.Hands -> result.hands.forEach { hand ->
                drawSkeleton(hand.map(::point), HandLandmarker.HAND_CONNECTIONS.map { it.start() to it.end() })
            }
            is MpResult.Poses -> result.poses.forEach { pose ->
                drawSkeleton(pose.map(::point), PoseLandmarker.POSE_LANDMARKS.map { it.start() to it.end() })
            }
            is MpResult.Classes, null -> Unit
        }
    }
}

private fun DrawScope.drawSkeleton(points: List<Offset>, connections: List<Pair<Int, Int>>) {
    connections.forEach { (a, b) ->
        drawLine(Color.Green, points[a], points[b], strokeWidth = 3.dp.toPx())
    }
    points.forEach { drawCircle(Color.Red, radius = 4.dp.toPx(), center = it) }
}

@Composable
private fun MpLabels(result: MpResult?, modifier: Modifier = Modifier) {
    val lines = when (result) {
        is MpResult.Classes -> result.labels.map { (name, score) -> "$name ${(score * 100).toInt()}%" }
        is MpResult.Faces -> result.expressions.map { (name, score) -> "$name ${(score * 100).toInt()}%" }
        is MpResult.Hands -> result.gestures.map { (name, score) -> "$name ${(score * 100).toInt()}%" }
        else -> emptyList()
    }
    if (lines.isEmpty()) return
    Column(modifier = modifier.background(Color.Black.copy(alpha = 0.6f)).padding(8.dp)) {
        lines.forEach { Text(text = it, color = Color.White) }
    }
}
