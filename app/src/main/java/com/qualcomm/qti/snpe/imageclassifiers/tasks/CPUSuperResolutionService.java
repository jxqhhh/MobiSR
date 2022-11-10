/*
 * Copyright (c) 2016-2021 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.snpe.imageclassifiers.tasks;

import android.app.IntentService;
import android.content.Intent;
import android.content.Context;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.SystemClock;

import com.qualcomm.qti.snpe.FloatTensor;
import com.qualcomm.qti.snpe.NeuralNetwork;
import com.qualcomm.qti.snpe.Tensor;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.nio.ByteBuffer;

public class CPUSuperResolutionService extends IntentService {

    private static final int FLOAT_SIZE = 4;

    final String mInputLayer;

    final String mOutputLayer;

    private FloatBuffer mMeanImage;

    long mJavaExecuteTime = -1;

    float[] loadRgbBitmapAsFloat(Bitmap image) {
        final int[] pixels = new int[image.getWidth() * image.getHeight()];
        image.getPixels(pixels, 0, image.getWidth(), 0, 0,
                image.getWidth(), image.getHeight());

        final float[] pixelsBatched = new float[pixels.length * 3];
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                final int idx = y * image.getWidth() + x;
                final int batchIdx = idx * 3;

                final float[] rgb = extractColorChannels(pixels[idx]);
                pixelsBatched[batchIdx]     = rgb[0];
                pixelsBatched[batchIdx + 1] = rgb[1];
                pixelsBatched[batchIdx + 2] = rgb[2];
            }
        }
        return pixelsBatched;
    }

    float[] loadGrayScaleBitmapAsFloat(Bitmap image) {
        final int[] pixels = new int[image.getWidth() * image.getHeight()];
        image.getPixels(pixels, 0, image.getWidth(), 0, 0,
                image.getWidth(), image.getHeight());

        final float[] pixelsBatched = new float[pixels.length];
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                final int idx = y * image.getWidth() + x;

                final int rgb = pixels[idx];
                final float b = ((rgb)       & 0xFF);
                final float g = ((rgb >>  8) & 0xFF);
                final float r = ((rgb >> 16) & 0xFF);
                float grayscale = (float) (r * 0.3 + g * 0.59 + b * 0.11);

                pixelsBatched[idx] = preProcess(grayscale);
            }
        }
        return pixelsBatched;
    }

    private float[] extractColorChannels(int pixel) {
        float b = ((pixel)       & 0xFF);
        float g = ((pixel >>  8) & 0xFF);
        float r = ((pixel >> 16) & 0xFF);
        return new float[] {preProcess(r), preProcess(g), preProcess(b)}; // 存疑，RCAN是用scipy.misc.imread读取的图片（mode参数为None），imread则是直接调用的PIl的open
    }

    private float preProcess(float original) {
        return original; // RCAN performs the pre-processing within the model
    }

    float getMin(float[] array) {
        float min = Float.MAX_VALUE;
        for (float value : array) {
            if (value < min) {
                min = value;
            }
        }
        return min;
    }

    float getMax(float[] array) {
        float max = Float.MIN_VALUE;
        for (float value : array) {
            if (value > max) {
                max = value;
            }
        }
        return max;
    }

    private static final String LOG_TAG = CPUSuperResolutionService.class.getSimpleName();
    private static final String ACTION_CPU = "CPU";

    public CPUSuperResolutionService() {
        super("CPUSuperResolutionService");
        Set<String> inputNames = mNeuralNetwork.getInputTensorsNames();
        Set<String> outputNames = mNeuralNetwork.getOutputTensorsNames();
        if (inputNames.size() != 1 || outputNames.size() != 1) {
            throw new IllegalStateException("Invalid network input and/or output tensors.");
        } else {
            mInputLayer = inputNames.iterator().next();
            mOutputLayer = outputNames.iterator().next();
        }
    }

    public static NeuralNetwork mNeuralNetwork;

    public void processBitmap(final Bitmap bitmap) {

        System.out.println("jxq 1");
        Bitmap result = null;
        System.out.println(mNeuralNetwork.getInputTensorsShapes().get(mInputLayer));
        final FloatTensor tensor = mNeuralNetwork.createFloatTensor(
                mNeuralNetwork.getInputTensorsShapes().get(mInputLayer));

        System.out.println("jxq 2");
        final int[] dimensions = tensor.getShape();
        final boolean isGrayScale = (dimensions[1] == 1);
        float[] rgbBitmapAsFloat;
        if (!isGrayScale) {
            rgbBitmapAsFloat = loadRgbBitmapAsFloat(bitmap);
        } else {
            rgbBitmapAsFloat = loadGrayScaleBitmapAsFloat(bitmap);
        }
        tensor.write(rgbBitmapAsFloat, 0, rgbBitmapAsFloat.length);

        final Map<String, FloatTensor> inputs = new HashMap<>();
        inputs.put(mInputLayer, tensor);

        System.out.println("jxq 3");
        final long javaExecuteStart = SystemClock.elapsedRealtime();

        System.out.println("jxq 4");
        final Map<String, FloatTensor> outputs = mNeuralNetwork.execute(inputs);

        System.out.println("jxq 5");
        final long javaExecuteEnd = SystemClock.elapsedRealtime();
        mJavaExecuteTime = javaExecuteEnd - javaExecuteStart;
        /*
        for (Map.Entry<String, FloatTensor> output : outputs.entrySet()) {
            if (output.getKey().equals(mOutputLayer)) {
                FloatTensor outputTensor = output.getValue();

                final float[] array = new float[outputTensor.getSize()];
                outputTensor.read(array, 0, array.length);

                final int[] pixels = new int[mImage.getWidth() * mImage.getHeight() * 4];
                int pixels_idx = 0;
                for (int i = 0; i < mImage.getHeight()*2; i ++) {
                    for (int j = 0; j < mImage.getWidth() * 2; j ++) {
                        int pixel = 0xf0 << 24;
                        for (int c = 0; c < 3; c ++){
                            int idx = 3 * (i * mImage.getWidth() * 2 + j) + c;
                            int offset = 8 * c;
                            pixel += (((int)array[idx]) & 0xff) << offset;
                        }
                        pixels[pixels_idx] = pixel;
                        pixels_idx ++;
                    }
                }
                // ARGB_8888: (A & 0xff) << 24 | (B & 0xff) << 16 | (G & 0xff) << 8 | (R & 0xff)
                result = Bitmap.createBitmap(pixels, mImage.getWidth()*2, mImage.getHeight()*2, Bitmap.Config.ARGB_8888);
            }
        }
        */

        releaseTensors(inputs, outputs);

        System.out.println("jxq 6");

        final long end = SystemClock.elapsedRealtime();
        System.out.println("End patch super-resolution at "+end);
    }


    @SafeVarargs
    private final void releaseTensors(Map<String, ? extends Tensor>... tensorMaps) {
        for (Map<String, ? extends Tensor> tensorMap: tensorMaps) {
            for (Tensor tensor: tensorMap.values()) {
                tensor.release();
            }
        }
    }

    public static void processPatch(Context context, Bitmap bitmap) {
        System.out.println("Process patch started");
        Intent intent = new Intent(context, CPUSuperResolutionService.class);
        intent.setAction(ACTION_CPU);
        Bundle bundle = new Bundle();
        bundle.putParcelable("patch", bitmap);// 序列化
        intent.putExtras(bundle);// 发送数据
        context.startService(intent);

        System.out.println("service started");
    }

    @Override
    protected void onHandleIntent(Intent intent) {

        System.out.println("Intent coming");
        if (intent != null) {
            final String action = intent.getAction();
            if (ACTION_CPU.equals(action)) {
                final Bitmap bitmap = intent.getParcelableExtra("patch");
                System.out.println("Process bitmap started");
                processBitmap(bitmap);
            }
        }
    }


}
