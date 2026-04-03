package com.example.esp32_test

import android.graphics.Bitmap
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.net.NetworkRequest
import android.net.wifi.WifiNetworkSpecifier
import android.os.*
import android.util.Log
import android.view.PixelCopy
import android.view.SurfaceView
import android.widget.Toast
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import com.example.esp32_test.databinding.ActivityMainBinding
import com.github.niqdev.mjpeg.DisplayMode
import com.github.niqdev.mjpeg.Mjpeg
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import okhttp3.*
import rx.Subscription
import rx.android.schedulers.AndroidSchedulers
import rx.subscriptions.CompositeSubscription
import java.io.IOException
import java.util.concurrent.atomic.AtomicInteger

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val disposables = CompositeSubscription()
    private lateinit var faceDetector: FaceDetector
    private val frameHandler = Handler(Looper.getMainLooper())
    private lateinit var frameRunnable: Runnable

    // We will initialize OkHttpClient once a network is available
    private var okHttpClient: OkHttpClient? = null
    private var esp32Network: Network? = null
    private lateinit var connectivityManager: ConnectivityManager
    private var networkCallback: ConnectivityManager.NetworkCallback? = null

    private var eyesClosedStartTime: Long = 0L
    private var hasSentBlinkCommand = false
    private var hasSentOffCommand = false

    // ESP32 AP details
    private val ESP32_SSID = "ESP32-CAM-AP" // IMPORTANT: Change this to your ESP32's SSID
    private val ESP32_PASSWORD = "fiwi@768FU$#Nooo" // IMPORTANT: Change this if your ESP32 has a password
    private val ESP32_LED_URL_BLINK = "http://192.168.4.1/led?state=blink"
    private val ESP32_LED_URL_OFF   = "http://192.168.4.1/led?state=off"
    private val ESP32_STREAM_URL    = "http://192.168.4.1:81/stream"

    private val EYE_CLOSED_PROB_THRESHOLD = 0.5f
    private val CLOSED_DURATION_MS        = 1000L
    private val FRAME_INTERVAL_MS         = 500L
    private val frameCount = AtomicInteger(0)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        connectivityManager = getSystemService(CONNECTIVITY_SERVICE) as ConnectivityManager

        // Configure ML Kit face detector
        val realTimeOpts = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
            .build()
        faceDetector = FaceDetection.getClient(realTimeOpts)

        binding.connectButton.setOnClickListener {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                connectToEsp32Wifi()
            } else {
                Toast.makeText(this, "Network binding requires Android 10+", Toast.LENGTH_LONG).show()
                // For older versions, the user must connect manually.
                // You could prompt them to connect to the ESP32 Wi-Fi here.
            }
        }

        frameRunnable = object : Runnable {
            override fun run() {
                captureFrameAndDetectEyes()
                frameHandler.postDelayed(this, FRAME_INTERVAL_MS)
            }
        }
    }

    @RequiresApi(Build.VERSION_CODES.Q)
    private fun connectToEsp32Wifi() {
        Toast.makeText(this, "Attempting to connect to ESP32...", Toast.LENGTH_SHORT).show()

        val specifier = WifiNetworkSpecifier.Builder()
            .setSsid(ESP32_SSID)
            .setWpa2Passphrase(ESP32_PASSWORD)
            .setIsHiddenSsid(true) //If the SSID of the Esp32 is hidden
            .build()

        val request = NetworkRequest.Builder()
            .addTransportType(NetworkCapabilities.TRANSPORT_WIFI)
            .setNetworkSpecifier(specifier)
            .build()

        networkCallback = object : ConnectivityManager.NetworkCallback() {
            override fun onAvailable(network: Network) {
                super.onAvailable(network)
                Log.d("ESP32", "ESP32 network available: $network")
                esp32Network = network

                // Bind the process to this network
                // This ensures all new connections use this network
                connectivityManager.bindProcessToNetwork(esp32Network)

                // Initialize OkHttpClient to use this specific network
                okHttpClient = OkHttpClient.Builder()
                    .socketFactory(network.socketFactory)
                    .build()

                runOnUiThread {
                    Toast.makeText(this@MainActivity, "Connected to ESP32 Network!", Toast.LENGTH_LONG).show()
                    setupAndStartMjpegStream()
                }
            }

            override fun onLost(network: Network) {
                super.onLost(network)
                Log.e("ESP32", "Lost connection to ESP32 network")
                runOnUiThread {
                    Toast.makeText(this@MainActivity, "Lost ESP32 connection", Toast.LENGTH_LONG).show()
                    binding.mjpegView.stopPlayback()
                    frameHandler.removeCallbacks(frameRunnable)
                    // Unbind the process from the network
                    connectivityManager.bindProcessToNetwork(null)
                    okHttpClient = null
                }
            }

            override fun onUnavailable() {
                super.onUnavailable()
                Log.e("ESP32", "Could not find ESP32 network")
                runOnUiThread {
                    Toast.makeText(this@MainActivity, "ESP32 network unavailable", Toast.LENGTH_LONG).show()
                }
            }
        }
        connectivityManager.requestNetwork(request, networkCallback!!)
    }


    override fun onPause() {
        super.onPause()
        binding.mjpegView.stopPlayback()
        frameHandler.removeCallbacks(frameRunnable)
    }

    override fun onDestroy() {
        super.onDestroy()
        disposables.clear()
        faceDetector.close()
        frameHandler.removeCallbacks(frameRunnable)
        // Release the network request
        networkCallback?.let { connectivityManager.unregisterNetworkCallback(it) }
        // Unbind the process from the network
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            connectivityManager.bindProcessToNetwork(null)
        }
    }

    private fun setupAndStartMjpegStream() {
        if (okHttpClient == null || esp32Network == null) {
            Toast.makeText(this, "Network not ready for streaming.", Toast.LENGTH_SHORT).show()
            return
        }

        binding.mjpegView.stopPlayback()
        eyesClosedStartTime = 0L
        hasSentBlinkCommand = false
        hasSentOffCommand = false
        frameCount.set(0)
        binding.fpsText.text = "FPS: N/A"

        // The Mjpeg library uses its own OkHttpClient, so we can't directly inject our
        // network-bound one. However, since we've bound the *entire process* to the
        // ESP32 network, any new HTTP client instance it creates should also use it.
        val sub: Subscription = Mjpeg.newInstance()
            .open(ESP32_STREAM_URL, 5)
            .doOnNext { frameCount.incrementAndGet() }
            .observeOn(AndroidSchedulers.mainThread())
            .subscribe({ stream ->
                binding.mjpegView.apply {
                    setSource(stream)
                    setDisplayMode(DisplayMode.BEST_FIT)
                    showFps(false)
                }
                reportFpsPeriodically()
                frameHandler.removeCallbacks(frameRunnable)
                frameHandler.postDelayed(frameRunnable, FRAME_INTERVAL_MS)
            }, { err ->
                Log.e("ESP32", "Stream error: ${err.message}")
                Toast.makeText(this, "Stream error: ${err.message}", Toast.LENGTH_LONG).show()
            })

        disposables.add(sub)
    }

    private fun reportFpsPeriodically() {
        val fpsHandler = Handler(Looper.getMainLooper())
        fpsHandler.post(object : Runnable {
            override fun run() {
                val count = frameCount.getAndSet(0)
                binding.fpsText.text = "FPS: $count"
                fpsHandler.postDelayed(this, 1000L)
            }
        })
    }

    private fun captureFrameAndDetectEyes() {
        val surfaceView = binding.mjpegView
        if (surfaceView.width == 0 || surfaceView.height == 0) return

        val bitmap = Bitmap.createBitmap(surfaceView.width, surfaceView.height, Bitmap.Config.ARGB_8888)
        val pixelCopyHandler = Handler(Looper.getMainLooper())
        PixelCopy.request(surfaceView, bitmap, { copyResult ->
            if (copyResult == PixelCopy.SUCCESS) {
                runFaceDetection(bitmap)
            } else {
                Log.e("ESP32", "PixelCopy failed")
                onEyesOpen()
            }
        }, pixelCopyHandler)
    }

    private fun runFaceDetection(frame: Bitmap) {
        val image = InputImage.fromBitmap(frame, 0)
        faceDetector.process(image)
            .addOnSuccessListener { faces ->
                if (faces.isNotEmpty()) {
                    binding.eyeStatusText.text = "Eyes detected"
                    processFirstFace(faces[0])
                } else {
                    binding.eyeStatusText.text = "No face detected"
                    onEyesOpen()
                }
            }
            .addOnFailureListener {
                Log.e("ESP32", "Face detection failed: ${it.message}")
                binding.eyeStatusText.text = "Face detection error"
                onEyesOpen()
            }
    }

    private fun processFirstFace(face: Face) {
        val leftProb  = face.leftEyeOpenProbability ?: 1.0f
        val rightProb = face.rightEyeOpenProbability ?: 1.0f
        val eyesClosed = (leftProb < EYE_CLOSED_PROB_THRESHOLD && rightProb < EYE_CLOSED_PROB_THRESHOLD)

        if (eyesClosed) {
            if (eyesClosedStartTime == 0L) eyesClosedStartTime = System.currentTimeMillis()
            if (System.currentTimeMillis() - eyesClosedStartTime >= CLOSED_DURATION_MS && !hasSentBlinkCommand) {
                sendLedCommand(ESP32_LED_URL_BLINK)
                hasSentBlinkCommand = true
                hasSentOffCommand = false
            }
        } else {
            onEyesOpen()
        }
    }

    private fun onEyesOpen() {
        eyesClosedStartTime = 0L
        if (hasSentBlinkCommand && !hasSentOffCommand) {
            sendLedCommand(ESP32_LED_URL_OFF)
            hasSentOffCommand = true
            hasSentBlinkCommand = false
        }
    }

    private fun sendLedCommand(url: String) {
        if (okHttpClient == null) {
            Log.e("ESP32", "Cannot send command, OkHttpClient not initialized.")
            return
        }
        val request = Request.Builder().url(url).get().build()
        okHttpClient!!.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                Log.e("ESP32", "LED command failed: ${e.message}")
            }
            override fun onResponse(call: Call, response: Response) {
                Log.d("ESP32", "LED command success: ${response.message}")
                response.close()
            }
        })
    }
}