package com.qualcomm.qti.snpe.imageclassifiers.tasks;

import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Pair;

import com.qualcomm.qti.snpe.FloatTensor;
import com.qualcomm.qti.snpe.NeuralNetwork;
import com.qualcomm.qti.snpe.Tensor;
import com.qualcomm.qti.snpe.imageclassifiers.Model;
import com.qualcomm.qti.snpe.imageclassifiers.ModelOverviewFragmentController;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

public class SuperResolutionWithFloatTensorTask extends AbstractSuperResolutionTask {

    private static final String LOG_TAG = SuperResolutionWithFloatTensorTask.class.getSimpleName();

    public SuperResolutionWithFloatTensorTask(ModelOverviewFragmentController controller,
                                            NeuralNetwork network, Bitmap image, Model model) {
        super(controller, network, image, model);
    }

    @Override
    protected Bitmap doInBackground(Bitmap... params) {

        Bitmap result = null;
        System.out.println(mNeuralNetwork.getInputTensorsShapes().get(mInputLayer));
        final FloatTensor tensor = mNeuralNetwork.createFloatTensor(
                mNeuralNetwork.getInputTensorsShapes().get(mInputLayer));

        loadMeanImageIfAvailable(mModel.meanImage, tensor.getSize());

        final int[] dimensions = tensor.getShape();
        final boolean isGrayScale = (dimensions[1] == 1);
        float[] rgbBitmapAsFloat;
        if (!isGrayScale) {
            rgbBitmapAsFloat = loadRgbBitmapAsFloat(mImage);
        } else {
            rgbBitmapAsFloat = loadGrayScaleBitmapAsFloat(mImage);
        }
        tensor.write(rgbBitmapAsFloat, 0, rgbBitmapAsFloat.length);

        final Map<String, FloatTensor> inputs = new HashMap<>();
        inputs.put(mInputLayer, tensor);

        final long javaExecuteStart = SystemClock.elapsedRealtime();
        final Map<String, FloatTensor> outputs = mNeuralNetwork.execute(inputs);
        final long javaExecuteEnd = SystemClock.elapsedRealtime();
        mJavaExecuteTime = javaExecuteEnd - javaExecuteStart;

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

        releaseTensors(inputs, outputs);

        return result;
    }

    @SafeVarargs
    private final void releaseTensors(Map<String, ? extends Tensor>... tensorMaps) {
        for (Map<String, ? extends Tensor> tensorMap: tensorMaps) {
            for (Tensor tensor: tensorMap.values()) {
                tensor.release();
            }
        }
    }
}
