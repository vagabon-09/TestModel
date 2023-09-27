package com.vagabon.testmodel;

import android.content.Intent;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.vagabon.testmodel.ml.Module;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;


public class MainActivity extends AppCompatActivity {
    private final String TAG = "MainActivityDebug";
    ImageView img;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Button button = findViewById(R.id.button);
        img = findViewById(R.id.img);
        button.setOnClickListener(v -> {
            Intent intent = new Intent(Intent.ACTION_PICK);
            intent.setType("image/*");
            startActivityForResult(intent, 100);
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (data != null && requestCode == 100) {
            Uri imageUri = data.getData();
            Bitmap bitmap = null;
            try {
                assert imageUri != null;
                bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageUri);
            } catch (Exception e) {
                Log.d(TAG, "onActivityResult: " + e.getMessage());
            }
            img.setImageBitmap(bitmap);
            int dimention = Math.min(bitmap.getWidth(), bitmap.getHeight());
            bitmap = ThumbnailUtils.extractThumbnail(bitmap, dimention, dimention);
            classifyImage(bitmap);
        }


    }


    private void classifyImage(Bitmap bitmap) {
        int imageSize = 224;
        try {
            Module model = Module.newInstance(MainActivity.this);

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, imageSize, imageSize, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());
            int[] intValues = new int[imageSize * imageSize];
//            bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
            int pixel = 0;
            for (int i = 0; i < imageSize; i++) {
                for (int j = 0; j < imageSize; j++) {
                    int val = intValues[pixel++];
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f/225));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f/225));
                    byteBuffer.putFloat((val & 0xFF) * (1.f/225));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Module.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            float[] confidences = outputFeature0.getFloatArray();
            int maxPos = 0;
            float maxConfidences = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidences) {
                    maxConfidences = confidences[i];
                    maxPos = i;
                }
            }
            Log.d(TAG, "classifyImage: index is -> " + maxPos);
            String[] arr = {"apple", "kiwi", "klemon", "lemon", "o'", "oranage", "orange", "pomegranate", "pomegranate", "tomato"};
            // Releases model resources if no longer used.
            try {
                Log.d(TAG, "classifyImage: " + arr[maxPos]);
            } catch (ArrayIndexOutOfBoundsException e) {
                Log.d(TAG, "classifyImage: 1 " + e.getMessage());
            }

            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }

    }


    private int getMax(float[] arr) {
        int maxIndex = 0;
        float maxVal = arr[0];

        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > maxVal) {
                maxVal = arr[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }
}