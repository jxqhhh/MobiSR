package com.qualcomm.qti.snpe.imageclassifiers.tasks;

import android.graphics.Bitmap;
import android.os.AsyncTask;
import android.util.Pair;

import com.qualcomm.qti.snpe.NeuralNetwork;
import com.qualcomm.qti.snpe.imageclassifiers.Model;
import com.qualcomm.qti.snpe.imageclassifiers.ModelOverviewFragmentController;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Set;

public abstract class AbstractSuperResolutionTask extends AsyncTask<Bitmap, Void, Bitmap> {

    private static final String LOG_TAG = AbstractClassifyImageTask.class.getSimpleName();

    private static final int FLOAT_SIZE = 4;

    final String mInputLayer;

    final String mOutputLayer;

    private final ModelOverviewFragmentController mController;

    final NeuralNetwork mNeuralNetwork;

    final Model mModel;

    final Bitmap mImage;

    private FloatBuffer mMeanImage;

    long mJavaExecuteTime = -1;

    AbstractSuperResolutionTask(ModelOverviewFragmentController controller,
                              NeuralNetwork network, Bitmap image, Model model) {
        mController = controller;
        mNeuralNetwork = network;
        mImage = image;
        mModel = model;

        Set<String> inputNames = mNeuralNetwork.getInputTensorsNames();
        Set<String> outputNames = mNeuralNetwork.getOutputTensorsNames();
        if (inputNames.size() != 1 || outputNames.size() != 1) {
            throw new IllegalStateException("Invalid network input and/or output tensors.");
        } else {
            mInputLayer = inputNames.iterator().next();
            mOutputLayer = outputNames.iterator().next();
        }

    }

    @Override
    protected void onPostExecute(Bitmap hr) {
        super.onPostExecute(hr);
        mController.onSuperResolutionResult(hr);
    }

    void loadMeanImageIfAvailable(File meanImage, final int imageSize) {
        ByteBuffer buffer = ByteBuffer.allocate(imageSize * FLOAT_SIZE)
                .order(ByteOrder.nativeOrder());
        if (!meanImage.exists()) {
            return;
        }
        FileInputStream fileInputStream = null;
        try {
            fileInputStream = new FileInputStream(meanImage);
            final byte[] chunk = new byte[1024];
            int read;
            while ((read = fileInputStream.read(chunk)) != -1) {
                buffer.put(chunk, 0, read);
            }
            buffer.flip();
        } catch (IOException e) {
            buffer = ByteBuffer.allocate(imageSize * FLOAT_SIZE);
        } finally {
            if (fileInputStream != null) {
                try {
                    fileInputStream.close();
                } catch (IOException e) {
                    // Do thing
                }
            }
        }
        mMeanImage = buffer.asFloatBuffer();
    }

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
        String modelName = mModel.name;

        float b = ((pixel)       & 0xFF);
        float g = ((pixel >>  8) & 0xFF);
        float r = ((pixel >> 16) & 0xFF);

        if (modelName.equals("inception_v3")) {
            return new float[] {preProcess(r), preProcess(g), preProcess(b)};
        } else if (modelName.equals("alexnet") && mMeanImage != null) {
            return new float[] {preProcess(b), preProcess(g), preProcess(r)};
        } else if (modelName.equals("googlenet") && mMeanImage != null) {
            return new float[] {preProcess(b), preProcess(g), preProcess(r)};
        } else if (modelName.equals("rcan_bix2")) {
            return new float[] {preProcess(r), preProcess(g), preProcess(b)}; // 存疑，RCAN是用scipy.misc.imread读取的图片（mode参数为None），imread则是直接调用的PIl的open
        } else {
            return new float[] {preProcess(r), preProcess(g), preProcess(b)};
        }
    }

    private float preProcess(float original) {
        String modelName = mModel.name;

        if (modelName.equals("inception_v3")) {
            return (original - 128) / 128;
        } else if (modelName.equals("alexnet") && mMeanImage != null) {
            return original - mMeanImage.get();
        } else if (modelName.equals("googlenet") && mMeanImage != null) {
            return original - mMeanImage.get();
        } else {
            return original; // RCAN performs the pre-processing within the model
        }
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
}
