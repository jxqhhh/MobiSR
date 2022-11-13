/*
 * Copyright (c) 2016-2018 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.snpe.imageclassifiers;

import android.app.Application;
import android.content.Context;
import android.graphics.Bitmap;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Handler;
import android.os.ResultReceiver;
import android.os.SystemClock;

import com.qualcomm.qti.snpe.NeuralNetwork;
import com.qualcomm.qti.snpe.SNPE;
import com.qualcomm.qti.snpe.imageclassifiers.tasks.CPUSuperResolutionService;
import com.qualcomm.qti.snpe.imageclassifiers.tasks.DSPSuperResolutionService;
import com.qualcomm.qti.snpe.imageclassifiers.tasks.GPUSuperResolutionService;
import com.qualcomm.qti.snpe.imageclassifiers.tasks.LoadImageTask;
import com.qualcomm.qti.snpe.imageclassifiers.tasks.LoadNetworkTask;

import java.io.File;
import java.lang.ref.SoftReference;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class ModelOverviewFragmentController extends AbstractViewController<ModelOverviewFragment> {

    public enum SupportedTensorFormat {
        FLOAT;
    }

    final float CPUDelay = 40;
    final float GPUDelay = 35;
    final float DSPDelay = 15;

    final public static int patch_width = 90;
    final public static int patch_height = 160;
    final public static int overlapping_size = 10;
    final float TV_threshold = 12;

    private final Map<String, SoftReference<Bitmap>> mBitmapCache;

    private final Model mModel;

    private final Application mApplication;

    private NeuralNetwork mNeuralNetwork;

    private LoadNetworkTask mLoadTaskModel1CPU;

    private LoadNetworkTask mLoadTaskModel1GPU;

    private LoadNetworkTask mLoadTaskModel2DSP;

    private NeuralNetwork.Runtime mRuntime;

    private final Context mContext;

    private SupportedTensorFormat mCurrentSelectedTensorFormat;

    private SupportedTensorFormat mNetworkTensorFormat;

    private boolean mUnsignedPD;

    public static int SuperResolutionServiceResultCode = 1;
    public static String SuperResolutionServiceResultKey = "patch";

    public ModelOverviewFragmentController(final Context context, final Application application, Model model, boolean unsignedPD) {
        mContext = context;
        mBitmapCache = new HashMap<>();
        mApplication = application;
        mModel = model;
        mUnsignedPD = unsignedPD;
    }

    @Override
    protected void onViewAttached(ModelOverviewFragment view) {
        view.setModelName(mModel.name);
        view.setSupportedRuntimes(getSupportedRuntimes());
        view.setSupportedTensorFormats(Arrays.asList(SupportedTensorFormat.values()));
        loadImageSamples(view);
    }

    private void loadImageSamples(ModelOverviewFragment view) {
        for (int i = 0; i < mModel.jpgImages.length; i++) {
            final File jpeg = mModel.jpgImages[i];
            final Bitmap cached = getCachedBitmap(jpeg);
            if (cached != null) {
                view.addSampleBitmap(cached);
            } else {
                final LoadImageTask task = new LoadImageTask(this, jpeg);
                task.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
            }
        }
    }

    private Bitmap getCachedBitmap(File jpeg) {
        final SoftReference<Bitmap> reference = mBitmapCache.get(jpeg.getAbsolutePath());
        if (reference != null) {
            final Bitmap bitmap = reference.get();
            if (bitmap != null) {
                return bitmap;
            }
        }
        return null;
    }

    private List<NeuralNetwork.Runtime> getSupportedRuntimes() {
        final List<NeuralNetwork.Runtime> result = new LinkedList<>();
        final SNPE.NeuralNetworkBuilder builder = new SNPE.NeuralNetworkBuilder(mApplication);
        NeuralNetwork.RuntimeCheckOption runtimeCheck = NeuralNetwork.RuntimeCheckOption.NORMAL_CHECK;
        if (mUnsignedPD){
            runtimeCheck = NeuralNetwork.RuntimeCheckOption.UNSIGNEDPD_CHECK;
        }
        builder.setRuntimeCheckOption(runtimeCheck);
        for (NeuralNetwork.Runtime runtime : NeuralNetwork.Runtime.values()) {
            if (builder.isRuntimeSupported(runtime)) {
                result.add(runtime);
            }
        }
        return result;
    }

    @Override
    protected void onViewDetached(ModelOverviewFragment view) {
        if (mNeuralNetwork != null) {
            mNeuralNetwork.release();
            mNeuralNetwork = null;
        }
    }

    public void onBitmapLoaded(File imageFile, Bitmap bitmap) {
        mBitmapCache.put(imageFile.getAbsolutePath(), new SoftReference<>(bitmap));
        if (isAttached()) {
            getView().addSampleBitmap(bitmap);
        }
    }

    public void onSuperResolutionResult(Bitmap bitmap, long javaExecuteTime) {
        if (isAttached()) {
            ModelOverviewFragment view = getView();
            view.addSampleBitmap(bitmap);
            view.setJavaExecuteStatistics(javaExecuteTime);
        }
    }

    public void onNetworkLoaded(NeuralNetwork neuralNetwork, final long loadTime) {
        System.out.println("load one");
    }

    public void onNetworkLoadFailed() {
        System.out.println("load failed");
    }

    private static float getTV (Bitmap patch) {
        float TV = 0;
        final int[] pixels = new int[patch.getWidth() * patch.getHeight()];
        patch.getPixels(pixels, 0, patch.getWidth(), 0, 0,
                patch.getWidth(), patch.getHeight());
        for (int y = 0; y < patch.getHeight(); y++) {
            for (int x = 0; x < patch.getWidth(); x++) {
                final int idx = y * patch.getWidth() + x;
                final float b = pixels[idx] & 0xFF;
                float g = ((pixels[idx] >>  8) & 0xFF);
                float r = ((pixels[idx] >> 16) & 0xFF);

                final float b1 = pixels[idx+1] & 0xFF;
                float g1 = ((pixels[idx+1] >>  8) & 0xFF);
                float r1 = ((pixels[idx+1] >> 16) & 0xFF);


                final float b2 = pixels[idx+patch.getWidth()] & 0xFF;
                float g2 = ((pixels[idx+patch.getWidth()] >>  8) & 0xFF);
                float r2 = ((pixels[idx+patch.getWidth()] >> 16) & 0xFF);

                TV += (Math.abs(b1-b)+Math.abs(g1-g)+Math.abs(r1-r)+Math.abs(b2-b)+Math.abs(g2-g)+Math.abs(r2-r))/(float)(patch.getWidth()*patch.getHeight());
            }
        }
        return TV;
    }

    public void superResolution(final Bitmap bitmap) {
        //if (mNeuralNetwork != null) { // TODO: judge if all the network models are loaded
            // TODO: 在这记录开始时间
            // TODO: divide into patches
            // TODO: call AsyncTask。每个AsyncTask是一个线程顺序执行，多个AysncTask类对应多个线程。
        Handler handler = new Handler();
        ResultReceiver resultReceiver = new ResultReceiver(handler){
            @Override
            protected void onReceiveResult(int resultCode, Bundle resultData) {
                if(resultCode==SuperResolutionServiceResultCode){
                    final Bitmap patch = resultData.getParcelable(SuperResolutionServiceResultKey);
                }else{
                    System.out.println("Error: unexpected result code");
                }
            }
        };
        final long start = SystemClock.elapsedRealtime();
        System.out.println("Start super-resolution at "+start);
        float timeCPU = 0;
        float timeGPU = 0;
        for (int i = 0; i < (int)(Math.floor((bitmap.getHeight()-overlapping_size)/(patch_height-overlapping_size))); i ++){
            for (int j = 0; j < (int)(Math.floor((bitmap.getWidth()-overlapping_size)/(patch_width-overlapping_size))); j ++){
                Bitmap patch = Bitmap.createBitmap(bitmap, j*(patch_width-overlapping_size), patch_width, i*(patch_height-overlapping_size), patch_height);
                float TV = getTV(patch);
                if (TV<TV_threshold){
                    if(timeCPU+CPUDelay<timeGPU+GPUDelay){
                        CPUSuperResolutionService.processPatch(mContext, bitmap, resultReceiver);
                        timeCPU += CPUDelay;
                    }else{
                        GPUSuperResolutionService.processPatch(mContext, bitmap);
                        timeGPU += GPUDelay;
                    }
                }else {
                    DSPSuperResolutionService.processPatch(mContext, bitmap);
                }
            }
        }
            // TODO: 利用ResultReceiver获得数据

            // TODO： 在callback里记录完成时间（每次callback给n+1，当n等于numberPatches说明搞定了）

        //} else {
        //    getView().displayModelNotLoaded();
        //}
    }



    public void setTargetRuntime(NeuralNetwork.Runtime runtime) {
        mRuntime = runtime;
    }

    public void setTensorFormat(SupportedTensorFormat format) {
        mCurrentSelectedTensorFormat = format;
    }

    public void loadNetwork() {
        if (isAttached()) {
            ModelOverviewFragment view = getView();
            view.setLoadingNetwork(true);
            view.setNetworkDimensions(null);
            view.setOutputLayersNames(new HashSet<String>());
            view.setModelVersion("");
            view.setModelLoadTime(-1);
            view.setJavaExecuteStatistics(-1);
            view.setClassificationHint();

            final NeuralNetwork neuralNetwork = mNeuralNetwork;
            if (neuralNetwork != null) {
                neuralNetwork.release();
                mNeuralNetwork = null;
            }

            mNetworkTensorFormat = mCurrentSelectedTensorFormat;
            mLoadTaskModel1CPU = new LoadNetworkTask(mApplication, this, mModel, NeuralNetwork.Runtime.CPU, SupportedTensorFormat.FLOAT, mUnsignedPD);
            mLoadTaskModel1GPU = new LoadNetworkTask(mApplication, this, mModel, NeuralNetwork.Runtime.GPU, SupportedTensorFormat.FLOAT, mUnsignedPD);
            mLoadTaskModel2DSP = new LoadNetworkTask(mApplication, this, mModel, NeuralNetwork.Runtime.DSP, SupportedTensorFormat.FLOAT, mUnsignedPD);
            mLoadTaskModel1CPU.executeOnExecutor(AsyncTask.SERIAL_EXECUTOR);
            mLoadTaskModel1GPU.executeOnExecutor(AsyncTask.SERIAL_EXECUTOR);
            mLoadTaskModel2DSP.executeOnExecutor(AsyncTask.SERIAL_EXECUTOR);
        }
    }

    private int[] getInputDimensions() {
        Set<String> inputNames = mNeuralNetwork.getInputTensorsNames();
        Iterator<String> iterator = inputNames.iterator();
        return iterator.hasNext() ? mNeuralNetwork.getInputTensorsShapes().get(iterator.next()) : null;
    }
}
