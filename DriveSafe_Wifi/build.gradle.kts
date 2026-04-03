plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.example.esp32_test"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.esp32_test"
        minSdk = 32
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }
    buildFeatures {
        viewBinding = true
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.16.0")
    implementation("androidx.appcompat:appcompat:1.7.1")
//    implementation("com.google.android.material:material:1.12.0")
    implementation("com.github.niqdev:ipcam-view:2.4.1")
    implementation("io.reactivex.rxjava2:rxjava:2.2.21")
    implementation("io.reactivex.rxjava2:rxandroid:2.1.1")

    implementation("com.google.android.gms:play-services-mlkit-face-detection:17.1.0")            // standalone ML Kit  :contentReference[oaicite:8]{index=8}
    implementation("com.squareup.okhttp3:okhttp:4.12.0")                // REST client  :contentReference[oaicite:9]{index=9}
    implementation("io.reactivex:rxjava:1.3.8")
    implementation("io.reactivex:rxandroid:1.2.1")


    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.activity)
    implementation(libs.androidx.constraintlayout)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
}