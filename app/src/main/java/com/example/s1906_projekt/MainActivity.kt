package com.example.s1906_projekt
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import com.example.s1906_projekt.ui.theme.S1906_projektTheme
import java.io.File
import java.io.FileOutputStream

class MainActivity : ComponentActivity() {
    private lateinit var imageView: ImageView
    private lateinit var resultView: TextView
    private val REQUEST_IMAGE_CAPTURE = 1
    private lateinit var module: Module
    private val classNames = arrayOf("bojownik","danio","glonojad","gupik_samica","gupik_samiec","gurami","molinezja","neon","pielegnica","skalar","welon")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        imageView = findViewById(R.id.imageView)
        resultView = findViewById(R.id.resultText)
        module = Module.load(assetFilePath(this, "fish_model.pt"))

        val button: Button = findViewById(R.id.captureButton)
        button.setOnClickListener {
            val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE)
        }

    }
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            val imageBitmap = data?.extras?.get("data") as Bitmap
            imageView.setImageBitmap(imageBitmap)

            val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                Bitmap.createScaledBitmap(imageBitmap, 299, 299, false),
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                TensorImageUtils.TORCHVISION_NORM_STD_RGB
            )

            val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()
            val rawScores = outputTensor.dataAsFloatArray
            val expScores = rawScores.map { Math.exp(it.toDouble()) }
            val sumExp = expScores.sum()
            val probs = expScores.map { (it / sumExp).toFloat() }
            val maxIndex = probs.indices.maxByOrNull { probs[it] } ?: -1
            val confidence = probs[maxIndex] * 100
            val confidenceStr = String.format("%.1f", confidence)
            resultView.text = "To jest: ${classNames[maxIndex]} - pewność: ($confidenceStr%)"
        }
        super.onActivityResult(requestCode, resultCode, data)
    }
}
fun assetFilePath(context: Context, assetName: String): String {
    val file = File(context.filesDir, assetName)
    if (!file.exists()) {
        context.assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
                outputStream.flush()
            }
        }
    }
    return file.absolutePath
}

@Composable
fun Greeting(name: String, modifier: Modifier = Modifier) {
    Text(
        text = "Hello $name!",
        modifier = modifier
    )
}

@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    S1906_projektTheme {
        Greeting("Android")
    }
}