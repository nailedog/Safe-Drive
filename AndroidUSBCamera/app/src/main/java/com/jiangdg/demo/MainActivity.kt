/*
MainActivity.kt - Simplified
*/
package com.jiangdg.demo

import android.Manifest.permission.*
import android.content.Intent
import android.os.Bundle
import android.os.PowerManager
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.PermissionChecker
import androidx.fragment.app.Fragment
import com.jiangdg.ausbc.utils.ToastUtils
import com.jiangdg.ausbc.utils.Utils
import com.jiangdg.demo.databinding.ActivityMainBinding

/**
 * Simplified MainActivity for USB camera eye detection
 */
class MainActivity : AppCompatActivity() {
    private var mWakeLock: PowerManager.WakeLock? = null
    private lateinit var viewBinding: ActivityMainBinding

    override fun onNewIntent(intent: Intent?) {
        super.onNewIntent(intent)
        if (intent?.action == "android.hardware.usb.action.USB_DEVICE_ATTACHED") {
            // A USB device was attached. Re-initialize the fragment to handle the new device.
            replaceDemoFragment(DemoFragment())
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)
        replaceDemoFragment(DemoFragment())
    }

    override fun onStart() {
        super.onStart()
        mWakeLock = Utils.wakeLock(this)
    }

    override fun onStop() {
        super.onStop()
        mWakeLock?.apply {
            Utils.wakeUnLock(this)
        }
    }

    private fun replaceDemoFragment(fragment: Fragment) {
        val hasCameraPermission = PermissionChecker.checkSelfPermission(this, CAMERA)
        val hasStoragePermission = PermissionChecker.checkSelfPermission(this, WRITE_EXTERNAL_STORAGE)

        if (hasCameraPermission != PermissionChecker.PERMISSION_GRANTED ||
            hasStoragePermission != PermissionChecker.PERMISSION_GRANTED) {
            if (ActivityCompat.shouldShowRequestPermissionRationale(this, CAMERA)) {
                ToastUtils.show(R.string.permission_tip)
            }
            ActivityCompat.requestPermissions(
                this,
                arrayOf(CAMERA, WRITE_EXTERNAL_STORAGE, RECORD_AUDIO),
                REQUEST_CAMERA
            )
            return
        }
        val transaction = supportFragmentManager.beginTransaction()
        transaction.replace(R.id.fragment_container, fragment)
        transaction.commitAllowingStateLoss()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        when (requestCode) {
            REQUEST_CAMERA -> {
                val hasCameraPermission = PermissionChecker.checkSelfPermission(this, CAMERA)
                if (hasCameraPermission == PermissionChecker.PERMISSION_DENIED) {
                    ToastUtils.show(R.string.permission_tip)
                    return
                }
                replaceDemoFragment(DemoFragment())
            }
        }
    }

    companion object {
        private const val REQUEST_CAMERA = 0
    }
}