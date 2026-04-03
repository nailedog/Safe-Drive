package com.jiangdg.demo

import android.annotation.SuppressLint
import android.app.AlertDialog
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.graphics.Bitmap
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import android.media.MediaPlayer
import android.media.ToneGenerator
import android.media.AudioManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.LayoutInflater
import android.view.TextureView
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.core.content.ContextCompat
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.jiangdg.ausbc.MultiCameraClient
import com.jiangdg.ausbc.base.CameraFragment
import com.jiangdg.ausbc.callback.ICameraStateCallBack
import com.jiangdg.demo.databinding.FragmentDemoBinding
import com.jiangdg.ausbc.utils.ToastUtils
import com.jiangdg.ausbc.utils.bus.BusKey
import com.jiangdg.ausbc.utils.bus.EventBus
import com.jiangdg.ausbc.widget.AspectRatioTextureView
import com.jiangdg.ausbc.widget.CaptureMediaView
import com.jiangdg.ausbc.widget.IAspectRatio

class DemoFragment : CameraFragment(), CaptureMediaView.OnViewClickListener {

    private lateinit var mViewBinding: FragmentDemoBinding
    private lateinit var faceDetector: FaceDetector
    private var toneGenerator: ToneGenerator? = null

    // Handlers for frame analysis and alarm
    private val analysisHandler = Handler(Looper.getMainLooper())
    private val alarmHandler = Handler(Looper.getMainLooper())
    private val flashHandler = Handler(Looper.getMainLooper())

    // Safe analysis runnable with proper error handling
    private val analysisRunnable = object : Runnable {
        override fun run() {
            try {
                if (!isCameraOpened()) {
                    // Camera is not open, don't try to analyze
                    return
                }

                val container = mViewBinding.cameraViewContainer
                if (container.childCount == 0) {
                    // No camera view yet, try again later
                    analysisHandler.postDelayed(this, 1000)
                    return
                }

                val textureView = container.getChildAt(0) as? TextureView
                if (textureView == null) {
                    // Not a texture view, try again later
                    analysisHandler.postDelayed(this, 1000)
                    return
                }

                val bitmap = textureView.getBitmap()
                if (bitmap != null && !bitmap.isRecycled) {
                    processFrame(bitmap)
                } else {
                    Log.w("DemoFragment", "Bitmap is null or recycled")
                }

                // Continue analysis
                analysisHandler.postDelayed(this, 500)

            } catch (e: Exception) {
                Log.e("DemoFragment", "Error in analysis runnable", e)
                // Try again after delay
                analysisHandler.postDelayed(this, 1000)
            }
        }
    }

    // Alarm system variables
    private var isTrackingActive = false
    private var eyeCloseThreshold: Float = 0.3f
    private var sleepTimeout: Long = 1000L
    private var useFlashlight: Boolean = true
    private var selectedSoundIndex = 0

    // Eye tracking state
    private var eyesClosedStartTime: Long? = null
    private var isAlarmActive = false
    private var isFlashingActive = false

    // Alarm runnables
    private var alarmRunnable: Runnable? = null
    private var flashRunnable: Runnable? = null

    override fun initData() {
        super.initData()
        EventBus.with<Int>(BusKey.KEY_FRAME_RATE).observe(this) { frameRate ->
            Log.d("DemoFragment", "Frame rate: $frameRate")
        }

        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
            .build()
        faceDetector = FaceDetection.getClient(options)

        // Initialize tone generator for alarm sound
        try {
            toneGenerator = ToneGenerator(AudioManager.STREAM_ALARM, 100)
        } catch (e: Exception) {
            Log.e("DemoFragment", "Error creating tone generator", e)
        }
    }

    override fun onCameraState(
        self: MultiCameraClient.ICamera,
        code: ICameraStateCallBack.State,
        msg: String?
    ) {
        when (code) {
            ICameraStateCallBack.State.OPENED -> handleCameraOpened()
            ICameraStateCallBack.State.CLOSED -> handleCameraClosed()
            ICameraStateCallBack.State.ERROR -> handleCameraError(msg)
        }
    }

    private fun handleCameraError(msg: String?) {
        ToastUtils.show("Camera error: $msg")
        analysisHandler.removeCallbacks(analysisRunnable)
        stopTracking()
    }

    private fun handleCameraClosed() {
        ToastUtils.show("Camera closed")
        analysisHandler.removeCallbacks(analysisRunnable)
        stopTracking()
    }

    private fun handleCameraOpened() {
        ToastUtils.show("Camera opened successfully")
        // Start eye detection after a delay to ensure camera view is ready

        val container = mViewBinding.cameraViewContainer
        if (container.childCount > 0) {
            val cameraView = container.getChildAt(0)
            cameraView.scaleX = 1.34f  // To fill the circular container with rectangle camera feed
            cameraView.scaleY = 1.34f
        }


        analysisHandler.postDelayed(analysisRunnable, 2000) // 2 second delay
    }

    override fun getCameraView(): IAspectRatio {
        return AspectRatioTextureView(requireContext())
    }

    override fun getCameraViewContainer(): ViewGroup {
        return mViewBinding.cameraViewContainer
    }

    override fun getRootView(inflater: LayoutInflater, container: ViewGroup?): View {
        mViewBinding = FragmentDemoBinding.inflate(inflater, container, false)
        setupUI()
        return mViewBinding.root
    }

    private fun setupUI() {
        setupJourneyButton()
        setupSoundButtons()
        setupFlashlightControl()
        setupSleepTimeoutControls()
        setupEyeThresholdSlider()
        updateTrackingButtonState()
        addUsbInfoButton()
    }

    private fun addUsbInfoButton() {
        try {
            val usbInfoButton = Button(requireContext()).apply {
                text = "USB Info"
                textSize = 12f
                setPadding(8, 8, 8, 8)
                setOnClickListener { showAllUsbDevices() }
            }

            (mViewBinding.root as ViewGroup).addView(usbInfoButton)

        } catch (e: Exception) {
            Log.e("DemoFragment", "Error adding USB info button", e)
        }
    }

    private fun setupJourneyButton() {
        mViewBinding.journeyButton.setOnClickListener {
            if (isTrackingActive) {
                stopTracking()
            } else {
                startTracking()
            }
        }
    }

    private fun setupSoundButtons() {
        val soundButtons = arrayOf(
            mViewBinding.soundButton1,
            mViewBinding.soundButton2,
            mViewBinding.soundButton3,
            mViewBinding.soundButton4
        )

        soundButtons.forEachIndexed { index, button ->
            button.setOnClickListener {
                selectSound(index)
                playTestAlarm() // Test the alarm sound
            }
        }

        updateSoundButtonSelection()
    }

    private fun setupFlashlightControl() {
        mViewBinding.flashlightSwitch.setOnCheckedChangeListener { _, isChecked ->
            useFlashlight = isChecked
            ToastUtils.show("Flashlight: ${if (isChecked) "ON" else "OFF"}")
        }
    }

    private fun setupSleepTimeoutControls() {
        updateTimeoutDisplay()

        mViewBinding.decreaseTimeoutBtn.setOnClickListener {
            if (sleepTimeout > 600L) {
                sleepTimeout -= 200L
                updateTimeoutDisplay()
//                ToastUtils.show("Sleep timeout: ${sleepTimeout}ms")
            }
        }

        mViewBinding.increaseTimeoutBtn.setOnClickListener {
            if (sleepTimeout < 2400L) {
                sleepTimeout += 200L
                updateTimeoutDisplay()
//                ToastUtils.show("Sleep timeout: ${sleepTimeout}ms")
            }
        }
    }

    private fun setupEyeThresholdSlider() {
        mViewBinding.sensitivitySeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                val thresholdPercent = 20 + progress
                eyeCloseThreshold = thresholdPercent / 100f
//                ToastUtils.show("Eye threshold: ${thresholdPercent}%")
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
    }

    private fun selectSound(index: Int) {
        selectedSoundIndex = index
        updateSoundButtonSelection()
        ToastUtils.show("Sound ${index + 1} selected")
    }

    private fun updateSoundButtonSelection() {
        try {
            val soundButtons = arrayOf(
                mViewBinding.soundButton1,
                mViewBinding.soundButton2,
                mViewBinding.soundButton3,
                mViewBinding.soundButton4
            )

            soundButtons.forEachIndexed { index, button ->
                if (index == selectedSoundIndex) {
                    button.setBackgroundResource(R.drawable.sound_button_selected_background)
                } else {
                    button.setBackgroundResource(R.drawable.sound_button_background)
                }
            }
        } catch (e: Exception) {
            ToastUtils.show("Error updating sound")
        }
    }

    private fun updateTimeoutDisplay() {
        val timeoutInSeconds = sleepTimeout / 1000f
        mViewBinding.timeoutValueTv.text = "${timeoutInSeconds} sec."
    }

    private fun startTracking() {
        if (!isCameraOpened()) {
            ToastUtils.show("Please wait for camera to open first")
            return
        }

        isTrackingActive = true
        updateTrackingButtonState()
        ToastUtils.show("Eye tracking active!")
    }

    private fun stopTracking() {
        isTrackingActive = false
        stopAlarm()
        resetEyeTrackingState()
        updateTrackingButtonState()
        ToastUtils.show("Eye tracking stopped")
    }

    private fun resetEyeTrackingState() {
        eyesClosedStartTime = null
    }

    private fun updateTrackingButtonState() {
        try {
            if (isTrackingActive) {
                mViewBinding.journeyButton.text = "Stop my journey"
                mViewBinding.journeyButton.setBackgroundResource(R.drawable.red_button_background)
            } else {
                mViewBinding.journeyButton.text = "Start my journey"
                mViewBinding.journeyButton.setBackgroundResource(R.drawable.green_button_background)
            }
        } catch (e: Exception) {
            ToastUtils.show("Error updating button state")
        }
    }

    private fun processFrame(bitmap: Bitmap) {
        try {
            val image = InputImage.fromBitmap(bitmap, 0)
            faceDetector.process(image)
                .addOnSuccessListener { faces: List<Face> ->
                    try {
                        var eyeOpennessText = "No face detected"
                        var currentEyeOpenness = 1.0f // Default to eyes open if no face

                        if (faces.isNotEmpty()) {
                            val face = faces[0]
                            val leftProb = face.leftEyeOpenProbability ?: -1f
                            val rightProb = face.rightEyeOpenProbability ?: -1f

                            var avgProb = 0f
                            var count = 0
                            if (leftProb >= 0) {
                                avgProb += leftProb
                                count++
                            }
                            if (rightProb >= 0) {
                                avgProb += rightProb
                                count++
                            }

                            if (count > 0) {
                                avgProb /= count
                                currentEyeOpenness = avgProb
                                val percentage = (avgProb * 100).toInt()
                                eyeOpennessText = "Eyes open: $percentage%"

                                // Add visual indicator when eyes are getting closed (only when tracking)
                                if (avgProb < eyeCloseThreshold && isTrackingActive) {
                                    eyeOpennessText += " ⚠️"
                                }
                            } else {
                                eyeOpennessText = "Eyes open: N/A"
                            }
                        }

                        // Update UI on main thread
                        analysisHandler.post {
                            try {
                                mViewBinding.eyeStatusTv.text = eyeOpennessText
                            } catch (e: Exception) {
                                Log.e("DemoFragment", "Error updating eye status text", e)
                            }
                        }

                        // Handle alarm logic ONLY when tracking is active
                        if (isTrackingActive) {
                            handleEyeClosedDetection(currentEyeOpenness)
                        }
                    } catch (e: Exception) {
                        Log.e("DemoFragment", "Error processing face detection success", e)
                    }
                }
                .addOnFailureListener {
                    try {
                        analysisHandler.post {
                            try {
                                mViewBinding.eyeStatusTv.text = "Detection error"
                            } catch (e: Exception) {
                                Log.e("DemoFragment", "Error updating eye status text on failure", e)
                            }
                        }
                        // Treat detection error as eyes open to avoid false alarms (only when tracking)
                        if (isTrackingActive) {
                            handleEyeClosedDetection(1.0f)
                        }
                    } catch (e: Exception) {
                        Log.e("DemoFragment", "Error processing face detection failure", e)
                    }
                }
        } catch (e: Exception) {
            Log.e("DemoFragment", "Error in processFrame", e)
        }
    }

    private fun handleEyeClosedDetection(eyeOpenness: Float) {
        if (!isTrackingActive) return

        try {
            if (eyeOpenness < eyeCloseThreshold) {
                // Eyes are closed
                if (eyesClosedStartTime == null) {
                    eyesClosedStartTime = System.currentTimeMillis()
                    Log.d("DemoFragment", "Eyes closed detected, starting timer...")
                } else {
                    val timeElapsed = System.currentTimeMillis() - eyesClosedStartTime!!
                    if (timeElapsed >= sleepTimeout && !isAlarmActive) {
                        Log.d("DemoFragment", "TRIGGERING ALARM! Eyes closed for ${timeElapsed}ms")
                        triggerAlarm()
                    }
                }
            } else {
                // Eyes are open
                if (eyesClosedStartTime != null) {
                    Log.d("DemoFragment", "Eyes opened, stopping alarm")
                    eyesClosedStartTime = null
                    if (isAlarmActive) {
                        stopAlarm()
                    }
                }
            }
        } catch (e: Exception) {
            Log.e("DemoFragment", "Error in handleEyeClosedDetection", e)
        }
    }

    private fun triggerAlarm() {
        if (isAlarmActive) return

        try {
            isAlarmActive = true

//            startAlarmSound()

            if (useFlashlight) {
                startFlashlight()
            }

            alarmHandler.postDelayed({
                if (isAlarmActive) {
                    stopAlarm()
                    ToastUtils.show("Alarm auto-stopped after 10 seconds")
                }
            }, 5000)
        } catch (e: Exception) {
            Log.e("DemoFragment", "Error triggering alarm", e)
        }
    }

    private fun stopAlarm() {
        if (!isAlarmActive) return

        try {
            isAlarmActive = false
            Log.d("DemoFragment", "Stopping alarm")

            stopAlarmSound()
            stopFlashlight()
            alarmHandler.removeCallbacksAndMessages(null)
        } catch (e: Exception) {
            Log.e("DemoFragment", "Error stopping alarm", e)
        }
    }

    private fun startAlarmSound() {
        try {
            // Play different tones based on selected sound
            val toneType = when (selectedSoundIndex) {
                0 -> ToneGenerator.TONE_CDMA_ALERT_CALL_GUARD
                1 -> ToneGenerator.TONE_CDMA_EMERGENCY_RINGBACK
                2 -> ToneGenerator.TONE_CDMA_HIGH_L
                3 -> ToneGenerator.TONE_CDMA_MED_L
                else -> ToneGenerator.TONE_CDMA_EMERGENCY_RINGBACK
            }

            alarmRunnable = object : Runnable {
                override fun run() {
                    if (isAlarmActive) {
                        toneGenerator?.startTone(toneType, 500) // Play for 500ms
                        alarmHandler.postDelayed(this, 600) // Repeat every 600ms
                    }
                }
            }
            alarmRunnable?.let { alarmHandler.post(it) }

        } catch (e: Exception) {
            Log.e("DemoFragment", "Error playing alarm sound", e)
        }
    }

    private fun stopAlarmSound() {
        try {
            alarmRunnable?.let { alarmHandler.removeCallbacks(it) }
            alarmRunnable = null
            toneGenerator?.stopTone()
        } catch (e: Exception) {
            Log.e("DemoFragment", "Error stopping alarm sound", e)
        }
    }

    private fun startFlashlight() {
        if (isFlashingActive) return

        try {
            isFlashingActive = true
            flashRunnable = object : Runnable {
                private var isOn = false
                override fun run() {
                    if (isFlashingActive && isAlarmActive) {
                        setTorchMode(isOn)
                        isOn = !isOn
                        flashHandler.postDelayed(this, 200) // Flash every 200ms
                    } else {
                        setTorchMode(false) // Turn off
                    }
                }
            }
            flashRunnable?.let { flashHandler.post(it) }
        } catch (e: Exception) {
            Log.e("DemoFragment", "Error starting flashlight", e)
        }
    }

    private fun stopFlashlight() {
        try {
            isFlashingActive = false
            flashRunnable?.let { flashHandler.removeCallbacks(it) }
            flashRunnable = null
            setTorchMode(false)
        } catch (e: Exception) {
            Log.e("DemoFragment", "Error stopping flashlight", e)
        }
    }

    private fun setTorchMode(on: Boolean) {
        try {
            val cameraManager = requireContext().getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val cameraId = cameraManager.cameraIdList.firstOrNull { id ->
                cameraManager.getCameraCharacteristics(id)
                    .get(CameraCharacteristics.FLASH_INFO_AVAILABLE) == true
            }
            cameraId?.let {
                cameraManager.setTorchMode(it, on)
            }
        } catch (e: Exception) {
            Log.e("DemoFragment", "Error controlling flashlight", e)
        }
    }

    private fun playTestAlarm() {
        try {
            val toneType = when (selectedSoundIndex) {
                0 -> ToneGenerator.TONE_CDMA_ALERT_CALL_GUARD
                1 -> ToneGenerator.TONE_CDMA_EMERGENCY_RINGBACK
                2 -> ToneGenerator.TONE_CDMA_HIGH_L
                3 -> ToneGenerator.TONE_CDMA_MED_L
                else -> ToneGenerator.TONE_CDMA_EMERGENCY_RINGBACK
            }
            toneGenerator?.startTone(toneType, 1000) // Play for 1 second
        } catch (e: Exception) {
            Log.e("DemoFragment", "Error playing test sound", e)
        }
    }

    private fun showAllUsbDevices() {
        try {
            val usbManager = requireContext().getSystemService(Context.USB_SERVICE) as UsbManager
            val deviceList = usbManager.deviceList

            if (deviceList.isEmpty()) {
                ToastUtils.show("No USB devices connected")
                return
            }

            val sb = StringBuilder()
            sb.append("All USB Devices:\n\n")

            var deviceCount = 0
            for ((_, device) in deviceList) {
                deviceCount++
                val deviceName = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP && !device.productName.isNullOrEmpty()) {
                    device.productName
                } else {
                    device.deviceName ?: "Unknown Device"
                }

                sb.append("Device $deviceCount:\n")
                sb.append("Name: $deviceName\n")
                sb.append("VID: 0x${String.format("%04X", device.vendorId)}\n")
                sb.append("PID: 0x${String.format("%04X", device.productId)}\n")
                sb.append("Device ID: ${device.deviceId}\n")
                sb.append("Path: ${device.deviceName}\n")

                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
                    device.manufacturerName?.let { sb.append("Manufacturer: $it\n") }
                    device.productName?.let { sb.append("Product: $it\n") }
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                        device.version.let { sb.append("Version: $it\n") }
                    }
                }
                sb.append("\n")
            }

            AlertDialog.Builder(requireContext())
                .setTitle("USB Device Information ($deviceCount devices)")
                .setMessage(sb.toString())
                .setPositiveButton("Copy to Clipboard") { _, _ ->
                    val clipboard = requireContext().getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                    val clip = ClipData.newPlainText("USB Device Info", sb.toString())
                    clipboard.setPrimaryClip(clip)
                    ToastUtils.show("Copied to clipboard")
                }
                .setNegativeButton("Close", null)
                .show()

        } catch (e: Exception) {
            Log.e("AllUsbDevices", "Error getting USB devices", e)
            ToastUtils.show("Error: ${e.message}")
        }
    }

    override fun onViewClick(mode: CaptureMediaView.CaptureMode?) {
        if (!isCameraOpened()) {
            ToastUtils.show("Camera not available!")
            return
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        try {
            stopTracking()
            toneGenerator?.release()
            analysisHandler.removeCallbacks(analysisRunnable)
            analysisHandler.removeCallbacksAndMessages(null)
            alarmHandler.removeCallbacksAndMessages(null)
            flashHandler.removeCallbacksAndMessages(null)
        } catch (e: Exception) {
            Log.e("DemoFragment", "Error in onDestroy", e)
        }
    }

    companion object {
        private const val TAG = "DemoFragment"
    }
}