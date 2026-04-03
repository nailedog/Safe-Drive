/*
SplashActivity.kt - Simplified
*/
package com.jiangdg.demo

import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.WindowManager
import androidx.appcompat.app.AppCompatActivity

/**
 * Simple splash activity for USB camera eye detection app
 */
class SplashActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN)
        setContentView(R.layout.activity_splash)
        startMainActivity()
    }

    private fun startMainActivity() {
        Handler(Looper.getMainLooper()).postDelayed({
            Intent(this, MainActivity::class.java).apply {
                startActivity(this)
                finish()
            }
        }, 500)
    }
}