/*
 * Copyright (c) 2016-2018 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.snpe.imageclassifiers.tasks;

import android.app.Application;
import android.os.AsyncTask;
import android.os.SystemClock;
import android.util.Log;

import com.qualcomm.qti.snpe.NeuralNetwork;
import com.qualcomm.qti.snpe.SNPE;
import com.qualcomm.qti.snpe.imageclassifiers.Model;
import com.qualcomm.qti.snpe.imageclassifiers.ModelOverviewFragmentController;
import com.qualcomm.qti.snpe.imageclassifiers.ModelOverviewFragmentController.SupportedTensorFormat;

import java.io.File;

public class LoadNetworkTask extends AsyncTask<File, Void, NeuralNetwork> {

    private static final String LOG_TAG = LoadNetworkTask.class.getSimpleName();

    private final ModelOverviewFragmentController mController;

    private final Model mModel;

    private final Application mApplication;

    private final NeuralNetwork.Runtime mTargetRuntime;

    private final SupportedTensorFormat mTensorFormat;

    private boolean mUnsignedPD;

    private long mLoadTime = -1;

    public LoadNetworkTask(final Application application,
                           final ModelOverviewFragmentController controller,
                           final Model model,
                           final NeuralNetwork.Runtime targetRuntime,
                           final SupportedTensorFormat tensorFormat,
                           boolean unsignedPD) {
        mApplication = application;
        mController = controller;
        mModel = model;
        mTargetRuntime = targetRuntime;
        mTensorFormat = tensorFormat;
        mUnsignedPD = unsignedPD;
    }

    @Override
    protected NeuralNetwork doInBackground(File... params) {
        NeuralNetwork network = null;
        try {
            final SNPE.NeuralNetworkBuilder builder= new SNPE.NeuralNetworkBuilder(mApplication)
                    .setDebugEnabled(false)
                    .setRuntimeOrder(mTargetRuntime)
                    .setCpuFallbackEnabled(true)
                    .setUseUserSuppliedBuffers(mTensorFormat != SupportedTensorFormat.FLOAT)
                    .setUnsignedPD(mUnsignedPD);
            if (mTargetRuntime.equals(NeuralNetwork.Runtime.DSP)){
                builder.setModel(mModel.file2);
            } else {
                builder.setModel(mModel.file);
            }
            if (mUnsignedPD){
                builder.setRuntimeCheckOption(NeuralNetwork.RuntimeCheckOption.UNSIGNEDPD_CHECK);
            }

            final long start = SystemClock.elapsedRealtime();
            network = builder.build();
            final long end = SystemClock.elapsedRealtime();

            if(mTargetRuntime.equals(NeuralNetwork.Runtime.CPU)){
                CPUSuperResolutionService.mNeuralNetwork = network;
            }else if(mTargetRuntime.equals(NeuralNetwork.Runtime.GPU)){
                GPUSuperResolutionService.mNeuralNetwork = network;
            }else if(mTargetRuntime.equals(NeuralNetwork.Runtime.DSP)){
                DSPSuperResolutionService.mNeuralNetwork = network;
            }else {
                throw new Exception("Not supported runtime");
            }
            mLoadTime = end - start;
        } catch (Exception e) {
            Log.e(LOG_TAG, e.getMessage(), e);
        }
        return network;
    }

    @Override
    protected void onPostExecute(NeuralNetwork neuralNetwork) {
        super.onPostExecute(neuralNetwork);
        if (neuralNetwork != null) {
            if (!isCancelled()) {
                mController.onNetworkLoaded(neuralNetwork, mLoadTime);
            } else {
                neuralNetwork.release();
            }
        } else {
            if (!isCancelled()) {
                mController.onNetworkLoadFailed();
            }
        }
    }
}
