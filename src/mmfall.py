# Cell 1
from google.colab import drive
drive.mount('/content/drive')

# Install Keras 2 compatibility layer for TF 2.16+
# This is needed because Colab ships TF 2.16+ with Keras 3,
# which breaks the standalone `from keras.*` imports and .h5 weight loading.
!pip install tf_keras
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"


# ----------------------------------------------------------------------------

# Cell 2


# ----------------------------------------------------------------------------

# Cell 3
# Author: Feng Jin. Contact info: fengjin@email.arizona.edu
import argparse, os
import matplotlib.pyplot as plt
import numpy as np
import random as rn
import tensorflow as tf
from keras import backend as K
from keras import optimizers
from keras.layers import Input, Dense, Flatten, Lambda, Concatenate, Reshape, \
    TimeDistributed, LSTM, RepeatVector, SimpleRNN, Activation
from keras.models import Model, load_model
from keras.callbacks import TensorBoard
from keras.losses import mean_squared_error as mse
from keras.utils import plot_model

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
        #   'figure.figsize': (19.2, 10.8),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

class data_preproc:
    def __init__(self):
        # One motion pattern is as (self.frames_per_pattern, self.points_per_frame, self.features_per_point)
        self.frames_per_pattern     = 10 # For 1 second for 10 fps radar rate
        self.points_per_frame       = 64 # We want to oversample every radar frame to 64 points while keeping the mean and variance the same
        self.features_per_point     = 4  # The radar can provides us (x, y, z, Doppler, RCS), but we only keep the first four feature, i.e. (x, y, z, Doppler)

        # Train and test data split ratio
        self.split_ratio            = 0.8

        # Rotation matrix due to tilt angle
        tilt_angle  = 10.0 # degrees
        self.height = 1.8   # meters
        self.rotation_matrix = np.array([[1.0, 0.0, 0.0],\
                                        [0.0, np.cos(np.deg2rad(tilt_angle)), np.sin(np.deg2rad(tilt_angle))],\
                                        [0.0, -np.sin(np.deg2rad(tilt_angle)), np.cos(np.deg2rad(tilt_angle))]])

    def load_bin(self, binfile_path, fortrain=True):
        # Record centroid history for analysis purpose
        centroidX_his = []
        centroidY_his = []
        centroidZ_his = []

        # Load the recorded bin file
        raw_pointcloud = np.load(binfile_path, allow_pickle=True)

        # Accumulated the motion patterns with (self.frames_per_pattern) frames
        total_pattern = []
        for idx in range(len(raw_pointcloud)-self.frames_per_pattern):
            total_pattern.append(raw_pointcloud[idx : (idx + self.frames_per_pattern)])

        # Original point vector in the .bin file:
        #   [frame number, point ID, target ID, \\ Basic information
        #   centroidX, centroidY, centroidZ, centroidVelocityX, centroidVelocityY, centroidVelocityZ, \\ Centorid information
        #   range, azimuth angle, elevation angle, Doppler, SNR, noise level] \\ Point information
        # Extract the feature vector (delta_x, delta_y, z, D, pointRCS) from the original point vector and do data oversampling proposed in the paper
        total_processed_pattern = []
        total_processed_no_norm = []
        for pattern in total_pattern:
            # Get the centroid information from the very first frame in a pattern and do coordiante transformation
            # As the centroid is already in the radar Cartesian coordinates, we just need to transfer it to the ground Cartesian coordinates
            centroidX, centroidY, centroidZ, centroidVx, centroidVy, centroidVz = pattern[0][0][3], pattern[0][0][4], pattern[0][0][5], pattern[0][0][6], pattern[0][0][7], pattern[0][0][8]
            results      = np.matmul(self.rotation_matrix, np.array([centroidX,centroidY,centroidZ]))
            centroidX    = results[0]
            centroidY    = results[1]
            centroidZ    = results[2] + self.height

            # Record the centroid history over time
            centroidX_his.append(centroidX)
            centroidY_his.append(centroidY)
            centroidZ_his.append(centroidZ)

            processed_pattern  = []
            for frame in pattern:
                processed_frame = []
                for point in frame:
                    # Get the original point information.
                    pointR, pointAZ, pointEL, pointD, pointSNR, pointNoise = point[9], point[10], point[11], point[12], point[13], point[14]

                    # Get the point's position in the Cartesian coord and then do coordiante transformation
                    pointX      = pointR*np.cos(pointEL)*np.sin(pointAZ)
                    pointY      = pointR*np.cos(pointEL)*np.cos(pointAZ)
                    pointZ      = pointR*np.sin(pointEL)
                    results     = np.matmul(self.rotation_matrix, np.array([pointX, pointY, pointZ]))
                    pointX      = results[0]
                    pointY      = results[1]
                    pointZ      = results[2] + self.height

                    # Subtract the point's position from the centroid in the very first frame in a motion pattern
                    delta_x     = pointX - centroidX
                    delta_y     = pointY - centroidY
                    delta_z     = pointZ
                    delta_D     = pointD
                    pointRCS    = 4*10*np.log10(pointR) + pointSNR*0.1 + pointNoise*0.1 # in dBsm

                    # Form the feature vector for each frame
                    feature_vector = [delta_x, delta_y, delta_z, delta_D, pointRCS]
                    processed_frame.append(feature_vector[0:self.features_per_point]) # Only keep 3D spatial info and the Doppler
                processed_pattern.append(processed_frame)
                # Do the data oversampling proposed in the paper
            processed_pattern_oversampled = self.proposed_oversampling(processed_pattern) # UPDATE - change this line to be done once per frame, not on every frame
            total_processed_pattern.append(processed_pattern_oversampled)

        total_processed_pattern_np = np.array(total_processed_pattern)

        # Train and test split
        split_idx   = int(total_processed_pattern_np.shape[0]*self.split_ratio)
        traindata   = total_processed_pattern_np[0:split_idx]
        testdata    = total_processed_pattern_np[split_idx:]

        if fortrain is True: # For training, need data split to obtain both training and testing dataset
            print("INFO: Total normal motion pattern data shape: " + str(total_processed_pattern_np.shape))
            print("INFO: Training motion pattern data shape" + str(traindata.shape))
            print("INFO: Testing motion pattern data shape" + str(testdata.shape))
            return traindata, testdata
        else: # For inference on anomaly dataset
            print("INFO: Total inference motion pattern data shape: " + str(total_processed_pattern_np.shape))
            return total_processed_pattern, centroidZ_his

    def proposed_oversampling(self, processed_pointcloud):
        # # Check the input
        # point_list = []
        # for frame in processed_pointcloud:
        #     point_list.extend(frame)
        # point_list_np  = np.array(point_list)
        # assert (point_list_np.shape[-1] == self.features_per_point), ("ERROR: Input processed_pointcloud has different feature length rather than %s!" %(self.features_per_point))

        # Do the data oversampling
        processed_pointcloud_oversampled = []
        for frame in processed_pointcloud:
            frame_np = np.array(frame)
            # Check if it's empty frame
            N = self.points_per_frame
            M = frame_np.shape[0]
            assert (M != 0), "ERROR: empty frame detected!"
            # Rescale and padding
            mean        = np.mean(frame_np, axis=0)
            sigma       = np.std(frame_np, axis=0)
            frame_np    = np.sqrt(N/M)*frame_np + mean - np.sqrt(N/M)*mean # Rescale
            frame_oversampled = frame_np.tolist()
            frame_oversampled.extend([mean]*(N-M)) # Padding with mean
            # # Check if mean and sigma keeps the same. Comment for saving time.
            # new_mean    = np.mean(np.array(frame_oversampled), axis=0)
            # new_sigma   = np.std(np.array(frame_oversampled), axis=0)
            # assert np.sum(np.abs(new_mean-mean))<1e-5, ("ERROR: Mean rescale and padding error!")
            # assert np.sum(np.abs(new_sigma-sigma))<1e-5, ("ERROR: Sigma rescale and padding error!")
            processed_pointcloud_oversampled.append(frame_oversampled)

        processed_pointcloud_oversampled_np = np.array(processed_pointcloud_oversampled)
        assert (processed_pointcloud_oversampled_np.shape[-2] == self.points_per_frame), ("ERROR: The new_frame_data has different number of points per frame rather than %s!" %(self.points_per_frame))
        assert (processed_pointcloud_oversampled_np.shape[-1] == self.features_per_point), ("ERROR: The new_frame_data has different feature length rather than %s!" %(self.features_per_point))

        return processed_pointcloud_oversampled_np

class autoencoder_mdl:
    def __init__(self, model_dir):
        self.model_dir = model_dir

    # --- NEW: Shared Helper Function to Build Model Structure ---
    def _build_HVRAE_architecture(self):
        # Model Parameters
        n_frames       = 10
        n_points       = 64
        n_features     = 4
        n_intermidiate = 64
        n_latentdim    = 16

        # Define input
        inputs          = Input(shape=(n_frames, n_points, n_features))
        input_flatten   = TimeDistributed(Flatten(data_format='channels_last'))(inputs)

        # VAE Encoder: q(z|X)
        input_flatten   = TimeDistributed(Dense(n_intermidiate, activation='tanh'))(input_flatten)
        Z_mean          = TimeDistributed(Dense(n_latentdim, activation=None), name='qzx_mean')(input_flatten)
        Z_log_var       = TimeDistributed(Dense(n_latentdim, activation=None), name='qzx_log_var')(input_flatten)

        # Sampling Function (The "Lambda" logic)
        def sampling(args):
            Z_mean, Z_log_var   = args

            # 1. Get the full shape as a tensor (works for dynamic & static dims)
            shape               = tf.shape(Z_mean)
            batch_size          = shape[0]
            n_frames            = shape[1]
            n_latentdim         = shape[2]

            # 2. Generate noise using standard TF
            epsilon             = tf.random.normal(shape=(batch_size, n_frames, n_latentdim), mean=0., stddev=1.0)

            # 3. Apply the reparameterization trick
            return Z_mean + tf.exp(0.5 * Z_log_var) * epsilon

        # Lambda Layer
        Z = Lambda(sampling, output_shape=(n_frames, n_latentdim))([Z_mean, Z_log_var])

        # RNN Autoencoder
        encoder_feature = SimpleRNN(n_latentdim, activation='tanh', return_sequences=False)(Z)
        decoder_feature = RepeatVector(n_frames)(encoder_feature)
        decoder_feature = SimpleRNN(n_latentdim, activation='tanh', return_sequences=True)(decoder_feature)
        decoder_feature = Lambda(lambda x: tf.reverse(x, axis=[-2]))(decoder_feature)

        # VAE Decoder: p(X|z)
        X_latent        = TimeDistributed(Dense(n_intermidiate, activation='tanh'))(decoder_feature)
        pXz_mean        = TimeDistributed(Dense(n_features, activation=None))(X_latent)
        pXz_logvar      = TimeDistributed(Dense(n_features, activation=None))(X_latent)

        # Reshape Output
        pXz     = Concatenate()([pXz_mean, pXz_logvar])
        pXz     = TimeDistributed(RepeatVector(n_points))(pXz)
        outputs = TimeDistributed(Reshape((n_points, n_features*2)))(pXz)

        # Return the uncompiled model
        return Model(inputs, outputs), Z_log_var, Z_mean

    # # Variational Recurrent Autoencoder (HVRAE)
    # def HVRAE_train(self, train_data, test_data):
    #     # In one motion pattern we have
    #     n_frames       = 10
    #     n_points       = 64
    #     n_features     = 4

    #     # Dimension is going down for encoding. Decoding is just a reflection of encoding.
    #     n_intermidiate    = 64
    #     n_latentdim       = 16

    #     # Define input
    #     inputs                  = Input(shape=(n_frames, n_points, n_features))
    #     input_flatten           = TimeDistributed(Flatten(None))(inputs)

    #     # VAE: q(z|X). Input: motion pattern. Output: mean and log(sigma^2) for q(z|X).
    #     input_flatten           = TimeDistributed(Dense(n_intermidiate, activation='tanh'))(input_flatten)
    #     Z_mean                  = TimeDistributed(Dense(n_latentdim, activation=None), name='qzx_mean')(input_flatten)
    #     Z_log_var               = TimeDistributed(Dense(n_latentdim, activation=None), name='qzx_log_var')(input_flatten)
    #     def sampling(args): # Instead of sampling from Q(z|X), sample epsilon = N(0,I), z = z_mean + sqrt(var) * epsilon
    #         Z_mean, Z_log_var   = args
    #         batch_size          = K.shape(Z_mean)[0]
    #         n_frames            = K.int_shape(Z_mean)[1]
    #         n_latentdim         = K.int_shape(Z_mean)[2]
    #         # For reproducibility, we set the seed=37
    #         epsilon             = K.random_normal(shape=(batch_size, n_frames, n_latentdim), mean=0., stddev=1.0, seed=None)
    #         Z                   = Z_mean + K.exp(0.5*Z_log_var) * epsilon # The reparameterization trick
    #         return  Z
    #     # VAE: sampling z ~ q(z|X) using reparameterization trick. Output: samples of z.
    #     Z                       = Lambda(sampling)([Z_mean, Z_log_var])

    #     # RNN Autoencoder. Output: reconstructed z.
    #     encoder_feature         = SimpleRNN(n_latentdim, activation='tanh', return_sequences=False)(Z)
    #     decoder_feature         = RepeatVector(n_frames)(encoder_feature)
    #     decoder_feature         = SimpleRNN(n_latentdim, activation='tanh', return_sequences=True)(decoder_feature)
    #     decoder_feature         = Lambda(lambda x: tf.reverse(x, axis=[-2]))(decoder_feature)

    #     # VAE: p(X|z). Output: mean and log(sigma^2) for p(X|z).
    #     X_latent                = TimeDistributed(Dense(n_intermidiate, activation='tanh'))(decoder_feature)
    #     pXz_mean                = TimeDistributed(Dense(n_features, activation=None))(X_latent)
    #     pXz_logvar              = TimeDistributed(Dense(n_features, activation=None))(X_latent)

    #     # Reshape the output. Output: (n_frames, n_points, n_features*2).
    #     # In each frame, every point has a corresponding mean vector with length of n_features and a log(sigma^2) vector with length of n_features.
    #     pXz                     = Concatenate()([pXz_mean, pXz_logvar])
    #     pXz                     = TimeDistributed(RepeatVector(n_points))(pXz)
    #     outputs                 = TimeDistributed(Reshape((n_points, n_features*2)))(pXz)

    #     # Build the model
    #     self.HVRAE_mdl = Model(inputs, outputs)
    #     print(self.HHVRAE_mdl.summary())
    def HVRAE_train(self, train_data, test_data):
        # Build the model architecture
        self.HVRAE_mdl, Z_log_var, Z_mean = self._build_HVRAE_architecture()
        print(self.HVRAE_mdl.summary())

        # Calculate HVRAE loss proposed in the paper
        def HVRAE_loss(y_true, y_pred):
            batch_size      = K.shape(y_true)[0]
            n_frames        = K.shape(y_true)[1]
            n_features      = K.shape(y_true)[-1]

            mean            = y_pred[:, :, :, :n_features]
            logvar          = y_pred[:, :, :, n_features:]
            var             = K.exp(logvar)

            y_true_reshape  = K.reshape(y_true, (batch_size, n_frames, -1))
            mean            = K.reshape(mean, (batch_size, n_frames, -1))
            var             = K.reshape(var, (batch_size, n_frames, -1))
            logvar          = K.reshape(logvar, (batch_size, n_frames, -1))

            # E[log_pXz] ~= log_pXz
            log_pXz         = K.square(y_true_reshape - mean)/var
            log_pXz         = K.sum(0.5*log_pXz, axis=-1)

            # KL divergence between q(z|x) and p(z)
            kl_loss         = -0.5 * K.sum(1 + Z_log_var - K.square(Z_mean) - K.exp(Z_log_var), axis=-1)

            # HVRAE loss is log_pXz + kl_loss
            HVRAE_loss        = K.mean(log_pXz + kl_loss) # Do mean over batches
            return HVRAE_loss

        # Define stochastic gradient descent optimizer Adam
        adam    = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        # Compile the model
        self.HVRAE_mdl.compile(optimizer=adam, loss=HVRAE_loss)

        # Train the model
        self.HVRAE_mdl.fit(train_data, train_data, # Train on the normal training dataset in an unsupervised way
                epochs=5,
                batch_size=8,
                shuffle=False,
                validation_data=(test_data, test_data), # Testing on the normal tesing dataset
                callbacks=[TensorBoard(log_dir=(self.model_dir + "/../model_history/HVRAE_online"))])
        self.HVRAE_mdl.save(self.model_dir + 'HVRAE_mdl_online.h5')
        # plot_model(self.HVRAE_mdl, show_shapes =True, to_file=self.model_dir+'HVRAE_mdl_online.png')
        print("INFO: Training is done!")
        print("*********************************************************************")

    def HVRAE_predict(self, inferencedata):
        inferencedata = np.array(inferencedata)
        K.clear_session()

        def sampling_predict(args): # Instead of sampling from Q(z|X), sample epsilon = N(0,I), z = z_mean + sqrt(var) * epsilon
            Z_mean, Z_log_var   = args

            # 1. Get the full shape as a tensor (works for dynamic & static dims)
            shape               = tf.shape(Z_mean)
            batch_size          = shape[0]
            n_frames            = shape[1]
            n_latentdim         = shape[2]

            # 2. Generate noise using standard TF
            epsilon             = tf.random.normal(shape=(batch_size, n_frames, n_latentdim), mean=0., stddev=1.0)

            # 3. Apply the reparameterization trick
            return Z_mean + tf.exp(0.5 * Z_log_var) * epsilon


        # Load saved model
        # model = load_model(self.model_dir + 'HVRAE_mdl.h5', compile = False, safe_mode=False, custom_objects={
        #                                                         'sampling': sampling_predict,
        #                                                         'tf': tf,
        #                                                         'Flatten': Flatten,
        #                                                         'Dense': Dense})
        # 1. Build the empty structure using your local code
        # (This sets up all the custom objects, layers, and lambdas automatically)
        model = self._build_HVRAE_architecture()[0]

        # 2. Fill it with the trained numbers
        model.load_weights(self.model_dir + 'HVRAE_mdl.h5')

        adam  = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        # Because we do not train the model, the loss function does not matter here.
        # Adding MSE as loss is omly for compiling the model. We can add any loss function here.
        # This is because our HVRAE loss function is customized function, we can not simply add it here.
        # We will define and call the HVRAE loss later.
        model.compile(optimizer=adam, loss=mse)
        print("INFO: Model loaded from " + self.model_dir + 'HVRAE_mdl.h5')

        get_z_mean_model    = Model(inputs=model.input, outputs=model.get_layer('qzx_mean').output)
        get_z_log_var_model = Model(inputs=model.input, outputs=model.get_layer('qzx_log_var').output)

        # Numpy version of HVRAE_loss function
        def HVRAE_loss(y_true, y_pred, Z_mean, Z_log_var):
            batch_size      = y_true.shape[0]
            n_frames        = y_true.shape[1]
            n_features      = y_true.shape[-1]

            mean            = y_pred[:, :, :, :n_features]
            logvar          = y_pred[:, :, :, n_features:]
            var             = np.exp(logvar)

            y_true_reshape  = np.reshape(y_true, (batch_size, n_frames, -1))
            mean            = np.reshape(mean, (batch_size, n_frames, -1))
            var             = np.reshape(var, (batch_size, n_frames, -1))
            logvar          = np.reshape(logvar, (batch_size, n_frames, -1))

            # E[log_pXz] ~= log_pXz
            # log_pXz       = K.square(y_true_reshape-mean)/var + logvar
            log_pXz         = np.square(y_true_reshape - mean)/var
            log_pXz         = np.sum(0.5*log_pXz, axis=-1)

            # KL divergence between q(z|x) and p(z)
            kl_loss         = -0.5 * np.sum(1 + Z_log_var - np.square(Z_mean) - np.exp(Z_log_var), axis=-1)

            # HVRAE loss is log_pXz + kl_loss
            HVRAE_loss        = np.mean(log_pXz + kl_loss) # Do mean over batches
            return HVRAE_loss

        print("INFO: Start to predict...")
        all_predictions     = model.predict(inferencedata, batch_size=64, verbose=0)
        all_z_mean          = get_z_mean_model.predict(inferencedata, batch_size=64, verbose=0)
        all_z_log_var       = get_z_log_var_model.predict(inferencedata, batch_size=64, verbose=0)

        n_features          = inferencedata.shape[-1]
        mean_pred           = all_predictions[:, :, :, :n_features]
        logvar_pred         = all_predictions[:, :, :, n_features:]
        var_pred            = np.exp(logvar_pred)

        y_reshape           = inferencedata.reshape(inferencedata.shape[0], inferencedata.shape[1], -1)
        mean_reshape        = mean_pred.reshape(mean_pred.shape[0], mean_pred.shape[1], -1)
        var_reshape         = var_pred.reshape(var_pred.shape[0], var_pred.shape[1], -1)

        log_pXz             = np.sum(0.5 * np.square(y_reshape - mean_reshape) / var_reshape, axis=-1)
        kl_loss             = -0.5 * np.sum(1 + all_z_log_var - np.square(all_z_mean) - np.exp(all_z_log_var), axis=-1)
        loss_per_pattern    = np.mean(log_pXz + kl_loss, axis=-1)
        loss_history        = loss_per_pattern.tolist()
        print("INFO: Prediction is done!")

        return loss_history


    def _build_HVRAE_SL_architecture(self):
        n_frames       = 10
        n_points       = 64
        n_features     = 4

        # Dimension is going down for encoding. Decoding is just a reflection of encoding.
        n_intermidiate    = 64
        n_latentdim       = 16

        # Define input
        inputs                  = Input(shape=(n_frames, n_points, n_features))
        input_flatten           = TimeDistributed(Flatten(None))(inputs)

        # VAE: q(z|X). Input: motion pattern. Output: mean and log(sigma^2) for q(z|X).
        input_flatten           = TimeDistributed(Dense(n_intermidiate, activation='tanh'))(input_flatten)
        Z_mean                  = TimeDistributed(Dense(n_latentdim, activation=None), name='qzx_mean')(input_flatten)
        Z_log_var               = TimeDistributed(Dense(n_latentdim, activation=None), name='qzx_log_var')(input_flatten)
        def sampling(args): # Instead of sampling from Q(z|X), sample epsilon = N(0,I), z = z_mean + sqrt(var) * epsilon
            Z_mean, Z_log_var   = args

            # 1. Get the full shape as a tensor (works for dynamic & static dims)
            shape               = tf.shape(Z_mean)
            batch_size          = shape[0]
            n_frames            = shape[1]
            n_latentdim         = shape[2]

            # 2. Generate noise using standard TF
            epsilon             = tf.random.normal(shape=(batch_size, n_frames, n_latentdim), mean=0., stddev=1.0)

            # 3. Apply the reparameterization trick
            return Z_mean + tf.exp(0.5 * Z_log_var) * epsilon

        # VAE: sampling z ~ q(z|X) using reparameterization trick. Output: samples of z.
        Z                       = Lambda(sampling)([Z_mean, Z_log_var])

        # RNN Autoencoder. Output: reconstructed z.
        encoder_feature         = SimpleRNN(n_latentdim, activation='tanh', return_sequences=False)(Z)
        decoder_feature         = RepeatVector(n_frames)(encoder_feature)
        decoder_feature         = SimpleRNN(n_latentdim, activation='tanh', return_sequences=True)(decoder_feature)
        decoder_feature         = Lambda(lambda x: tf.reverse(x, axis=[-2]))(decoder_feature)

        # VAE: p(X|z). Output: mean for p(X|z).
        X_latent                = TimeDistributed(Dense(n_intermidiate, activation='tanh'))(decoder_feature)
        pXz_mean                = TimeDistributed(Dense(n_points * n_features, activation=None))(X_latent)

        # Reshape the output. Output: (n_frames, n_points, n_features).
        # In each frame, every point has a corresponding mean vector with length of n_features.
        outputs                 = TimeDistributed(Reshape((n_points, n_features)))(pXz_mean)
        return Model(inputs, outputs), Z_log_var, Z_mean

    # Baseline #1: HVRAE_SL with simplified loss function
    def HVRAE_SL_train(self, train_data, test_data):
        # In one motion pattern we have

        # Build the model
        self.HVRAE_SL_mdl, Z_log_var, Z_mean = self._build_HVRAE_SL_architecture()
        print(self.HVRAE_SL_mdl.summary())

        # Calculate HVRAE loss proposed in the paper
        def HVRAE_loss(y_true, y_pred):
            batch_size      = K.shape(y_true)[0]
            n_frames        = K.shape(y_true)[1]
            n_features      = K.shape(y_true)[-1]

            mean            = y_pred

            y_true_reshape  = K.reshape(y_true, (batch_size, n_frames, -1))
            mean            = K.reshape(mean, (batch_size, n_frames, -1))

            # E[log_pXz] ~= log_pXz
            # In this case, the loss is MSE
            log_pXz         = K.square(y_true_reshape - mean)
            log_pXz         = K.sum(0.5*log_pXz, axis=-1)
            log_pXz         = mse(y_true_reshape, mean)

            # KL divergence between q(z|x) and p(z)
            kl_loss         = -0.5 * K.sum(1 + Z_log_var - K.square(Z_mean) - K.exp(Z_log_var), axis=-1)

            # HVRAE loss is log_pXz + kl_loss
            HVRAE_loss        = K.mean(log_pXz + kl_loss) # Do mean over batches
            return HVRAE_loss

        # Define stochastic gradient descent optimizer Adam
        adam    = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        # Compile the model
        self.HVRAE_SL_mdl.compile(optimizer=adam, loss=HVRAE_loss)

        # Train the model
        self.HVRAE_SL_mdl.fit(train_data, train_data, # Train on the normal training dataset in an unsupervised way
                epochs=5,
                batch_size=8,
                shuffle=False,
                validation_data=(test_data, test_data), # Testing on the normal tesing dataset
                callbacks=[TensorBoard(log_dir=(self.model_dir + "/../model_history/HVRAE_SL_online"))])
        self.HVRAE_SL_mdl.save(self.model_dir + 'HVRAE_SL_mdl_online.h5')
        # plot_model(self.HVRAE_SL_mdl, show_shapes =True, to_file=self.model_dir+'HVRAE_SL_mdl_online.png')
        print("INFO: Training is done!")
        print("*********************************************************************")

    def HVRAE_SL_predict(self, inferencedata):
        inferencedata = np.array(inferencedata)
        K.clear_session()

        def sampling_predict(args): # Instead of sampling from Q(z|X), sample epsilon = N(0,I), z = z_mean + sqrt(var) * epsilon
            Z_mean, Z_log_var   = args

            # 1. Get the full shape as a tensor (works for dynamic & static dims)
            shape               = tf.shape(Z_mean)
            batch_size          = shape[0]
            n_frames            = shape[1]
            n_latentdim         = shape[2]

            # 2. Generate noise using standard TF
            epsilon             = tf.random.normal(shape=(batch_size, n_frames, n_latentdim), mean=0., stddev=1.0)

            # 3. Apply the reparameterization trick
            return Z_mean + tf.exp(0.5 * Z_log_var) * epsilon

            # Z_mean, Z_log_var   = args
            # batch_size          = K.shape(Z_mean)[0]
            # n_frames            = K.int_shape(Z_mean)[1]
            # n_latentdim         = K.int_shape(Z_mean)[2]
            # # For reproducibility, we set the seed=37
            # epsilon             = K.random_normal(shape=(batch_size, n_frames, n_latentdim), mean=0., stddev=1.0, seed=None)
            # Z                   = Z_mean + K.exp(0.5*Z_log_var) * epsilon # The reparameterization trick
            # return  Z

        # Load the saved model
        # model = load_model(self.model_dir + 'HVRAE_SL_mdl.h5', compile = False, custom_objects={'sampling': sampling_predict, 'tf': tf})

        # 1. Build the empty structure using your local code
        # (This sets up all the custom objects, layers, and lambdas automatically)
        model = self._build_HVRAE_SL_architecture()[0]

        # 2. Fill it with the trained numbers
        model.load_weights(self.model_dir + 'HVRAE_SL_mdl.h5')
        adam  = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        # Because we do not train the model, the loss function does not matter here.
        # Adding MSE as loss is omly for compiling the model. We can add any loss function here.
        # This is because our HVRAE loss function is customized function, we can not simply add it here.
        # We will define and call the HVRAE loss later.
        model.compile(optimizer=adam, loss=mse)
        print("INFO: Model loaded from " + self.model_dir + 'HVRAE_SL_mdl.h5')

        get_z_mean_model    = Model(inputs=model.input, outputs=model.get_layer('qzx_mean').output)
        get_z_log_var_model = Model(inputs=model.input, outputs=model.get_layer('qzx_log_var').output)

        # Numpy version of HVRAE_loss function
        def HVRAE_loss(y_true, y_pred, Z_mean, Z_log_var):
            batch_size      = y_true.shape[0]
            n_frames        = y_true.shape[1]
            n_features      = y_true.shape[-1]

            mean            = y_pred

            y_true_reshape  = np.reshape(y_true, (batch_size, n_frames, -1))
            mean            = np.reshape(mean, (batch_size, n_frames, -1))

            # E[log_pXz] ~= log_pXz
            # In this case, the loss is MSE
            # log_pXz         = np.square(y_true_reshape - mean)
            # log_pXz         = np.sum(0.5*log_pXz, axis=-1)
            log_pXz         = (np.square(y_true_reshape - mean)).mean(axis=-1)

            # KL divergence between q(z|x) and p(z)
            kl_loss         = -0.5 * np.sum(1 + Z_log_var - np.square(Z_mean) - np.exp(Z_log_var), axis=-1)

            # HVRAE loss is log_pXz + kl_loss
            HVRAE_loss        = np.mean(log_pXz + kl_loss) # Do mean over batches
            return HVRAE_loss

        print("INFO: Start to predict...")
        all_predictions     = model.predict(inferencedata, batch_size=64, verbose=0)
        all_z_mean          = get_z_mean_model.predict(inferencedata, batch_size=64, verbose=0)
        all_z_log_var       = get_z_log_var_model.predict(inferencedata, batch_size=64, verbose=0)

        y_reshape           = inferencedata.reshape(inferencedata.shape[0], inferencedata.shape[1], -1)
        mean_reshape        = all_predictions.reshape(all_predictions.shape[0], all_predictions.shape[1], -1)

        log_pXz             = np.mean(np.square(y_reshape - mean_reshape), axis=-1)
        kl_loss             = -0.5 * np.sum(1 + all_z_log_var - np.square(all_z_mean) - np.exp(all_z_log_var), axis=-1)
        loss_per_pattern    = np.mean(log_pXz + kl_loss, axis=-1)
        loss_history        = loss_per_pattern.tolist()
        print("INFO: Prediction is done!")

        return loss_history


    def _build_RAE_architecture(self):
        # In one motion pattern we have
        n_frames       = 10
        n_points       = 64
        n_features     = 4

        # Dimension is going down for encoding. Decoding is just a reflection of encoding.
        n_intermidiate    = 64
        n_latentdim       = 16

        # Define input
        inputs                  = Input(shape=(n_frames, n_points, n_features))
        input_flatten           = TimeDistributed(Flatten(None))(inputs)

        # Embedding or feature compression
        encoder_feature         = TimeDistributed(Dense(n_intermidiate, activation='tanh'))(input_flatten)
        encoder_feature         = TimeDistributed(Dense(n_latentdim, activation='tanh'))(encoder_feature)

        # RNN Autoencoder.
        encoder_feature         = SimpleRNN(n_latentdim, activation='tanh', return_sequences=False)(encoder_feature)
        decoder_feature         = RepeatVector(n_frames)(encoder_feature)
        decoder_feature         = SimpleRNN(n_latentdim, activation='tanh', return_sequences=True)(decoder_feature)
        decoder_feature         = Lambda(lambda x: tf.reverse(x, axis=[-2]))(decoder_feature)

        # Embedding or feature compression
        decoder_feature         = TimeDistributed(Dense(n_intermidiate, activation='tanh'))(decoder_feature)
        decoder_feature         = TimeDistributed(Dense(n_points*n_features, activation='tanh'))(decoder_feature)

        # Reshape
        outputs                 = TimeDistributed(Reshape((n_points, n_features)))(decoder_feature)
        return Model(inputs, outputs)

    # Baseline #2: Recurrent AE
    def RAE_train(self, train_data, test_data):


        # Build the model
        self.RAE_mdl = self._build_RAE_architecture()
        print(self.RAE_mdl.summary())

        # Define stochastic gradient descent optimizer Adam
        adam    = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        # Compile the model
        self.RAE_mdl.compile(optimizer=adam, loss=mse)

        # Train the model
        self.RAE_mdl.fit(train_data, train_data, # Train on the normal training dataset in an unsupervised way
                epochs=5,
                batch_size=8,
                shuffle=False,
                validation_data=(test_data, test_data), # Testing on the normal tesing dataset
                callbacks=[TensorBoard(log_dir=(self.model_dir + "/../model_history/RAE_online"))])
        self.RAE_mdl.save(self.model_dir + 'RAE_mdl_online.h5')
        # plot_model(self.RAE_mdl, show_shapes =True, to_file=self.model_dir+'RAE_mdl_online.png')
        print("INFO: Training is done!")
        print("*********************************************************************")

    def RAE_predict(self, inferencedata):
        inferencedata = np.array(inferencedata)
        K.clear_session()

        # Load the saved model
        # model = load_model(self.model_dir + 'RAE_mdl.h5', compile = True, custom_objects={'tf': tf})
        model = self._build_RAE_architecture()

        # 2. Fill it with the trained numbers
        model.load_weights(self.model_dir + 'RAE_mdl.h5')
        model.compile(loss=mse)

        # plot_model(model, show_shapes =True, to_file=self.model_dir+'RAE_model.png')
        print("INFO: Start to predict...")
        all_predictions     = model.predict(inferencedata, batch_size=64, verbose=0)
        mse_per_point       = np.mean(np.square(inferencedata - all_predictions), axis=-1)
        mse_per_pattern     = np.mean(mse_per_point, axis=(1, 2))
        loss_history        = mse_per_pattern.tolist()
        print("INFO: Prediction is done!")

        return loss_history




# ----------------------------------------------------------------------------

# Cell 4
class compute_metric:
    def __init__(self):
        pass

    def detect_falls(self, loss_history, centroidZ_history, threshold):
        assert len(loss_history) == len(centroidZ_history), "ERROR: The length of loss history is different than the length of centroidZ history!"
        seq_len                 = len(loss_history)
        win_len                 = 20 # Detection window length on account of 2 seconds for 10 fps radar rate
        centroidZ_dropthres     = 0.6
        i                       = int(win_len/2)
        detected_falls_idx      = []
        # Firstly, detect the fall centers based on the centroidZ drop
        while i < (seq_len - win_len/2):
            detection_window_middle  = i
            detection_window_lf_edge = int(detection_window_middle - win_len/2)
            detection_window_rh_edge = int(detection_window_middle + win_len/2)
            # Search the centroidZ drop
            if centroidZ_history[detection_window_lf_edge] - centroidZ_history[detection_window_rh_edge] >= centroidZ_dropthres:
                detected_falls_idx.append(int(detection_window_middle))
            i += 1

        # Secondly, if a sequence of fall happen within a window less than win_len, we combine these falls into one fall centered at the middle of this sequence
        i = 0
        processed_detected_falls_idx = []
        while i < len(detected_falls_idx):
            j = i
            while True:
                if j == len(detected_falls_idx):
                    break
                if detected_falls_idx[j] - detected_falls_idx[i] > win_len:
                    break
                j += 1
            processed_detected_falls_idx.append(int((detected_falls_idx[i] + detected_falls_idx[j-1])/2))
            i = j

        # Thirdly, find id there is an anomaly level (or loss history) spike in the detection window
        ones_idx                    = np.argwhere(np.array(loss_history)>=threshold).flatten()
        fall_binseq                 = np.zeros(seq_len)
        fall_binseq[ones_idx]       = 1
        final_detected_falls_idx    = []
        i = 0
        while i < len(processed_detected_falls_idx):
            detection_window_middle  = int(processed_detected_falls_idx[i])
            detection_window_lf_edge = int(detection_window_middle - win_len/2)
            detection_window_rh_edge = int(detection_window_middle + win_len/2)
            if 1 in fall_binseq[detection_window_lf_edge:detection_window_rh_edge]:
                final_detected_falls_idx.append(processed_detected_falls_idx[i])
            i += 1

        return final_detected_falls_idx, len(processed_detected_falls_idx)

    def find_tpfpfn(self, detected_falls_idx, gt_falls_idx):
        n_detected_falls    = len(detected_falls_idx)
        falls_tp            = []
        falls_fp            = []
        falls_fn            = list(gt_falls_idx)
        win_len             = 20
        for i in range(n_detected_falls):
            n_gt_falls      = len(falls_fn)
            j               = 0
            while j < n_gt_falls:
                # Find a gt fall index whose window covers the detected fall index, so it's true positive
                if int(falls_fn[j]-win_len/2) <= detected_falls_idx[i] <= int(falls_fn[j]+win_len/2):
                    # Remove the true positive from the gt_falls_idx list, finally only false negative remains
                    falls_fn.pop(j)
                    falls_tp.append(i)
                    break
                j += 1
            # Dn not find a gt fall index whose window covers the detected fall index, so it's false positive
            if j == n_gt_falls:
                falls_fp.append(i)

        return falls_tp, falls_fp, falls_fn

    def cal_roc(self, loss_history, centroidZ_history, gt_falls_idx):
        n_gt_falls = len(gt_falls_idx)
        print("How many falls?", n_gt_falls)
        tpr, fpr = [], []
        for threshold in np.arange(0.0, 10.0, 0.1):
            detected_falls_idx, _           = self.detect_falls(loss_history, centroidZ_history, threshold)
            falls_tp, falls_fp, falls_fn    = self.find_tpfpfn(detected_falls_idx, gt_falls_idx)
            # Save the true positve rate for this threshold.
            if threshold == 0.6:
                print(f"---- regular calculator ----")
                print(f"\tfalls_tp: {falls_tp}")
                print(f"\tfalls_fp: {falls_fp}")
                print(f"\tfalls_fn: {falls_fn}")
                print("-----------------------------")
            tpr.append(len(falls_tp)/n_gt_falls)
            # Save the number of false positve, or missed fall detection, for this threshold
            fpr.append(len(falls_fp))
        return tpr, fpr


class compute_metric_early_predict:
    def __init__(self, prediction_gap, lower_bound=5):
        self.prediction_gap = prediction_gap
        self.lower_bound = lower_bound

    def detect_falls(self, loss_history, centroidZ_history, threshold, centroidZ_thr=0.6):
        assert len(loss_history) == len(centroidZ_history), "ERROR: The length of loss history is different than the length of centroidZ history!"
        seq_len                 = len(loss_history)
        win_len                 = 10
        centroidZ_dropthres     = centroidZ_thr
        i                       = win_len - 1
        detected_falls_idx      = []
        while i < seq_len:
            start_win = i - (win_len - 1)
            end_win = i
            if centroidZ_history[start_win] - centroidZ_history[end_win] >= centroidZ_dropthres:
                detected_falls_idx.append(int(end_win))
            i += 1


        ones_idx                    = np.argwhere(np.array(loss_history) >= threshold).flatten()
        fall_binseq                 = np.zeros(seq_len)
        fall_binseq[ones_idx]       = 1
        final_detected_falls_idx    = []
        for det in detected_falls_idx:
            detection_window_early_buffer = self.lower_bound
            detection_window_start = max(0, det - (win_len - 1) - detection_window_early_buffer)
            detection_window_end = det - (win_len - 1)
            if 1 in fall_binseq[detection_window_start:detection_window_end + 1]:
                final_detected_falls_idx.append(det)


        return final_detected_falls_idx, len(detected_falls_idx)

    def find_tpfpfn(self, detected_falls_idx, gt_falls_predict_idx_firsts):
        cluster_win_len = 20
        # Step 1: Cluster raw detections (store actual detection values per cluster)
        clustered_falls = []
        i = 0
        while i < len(detected_falls_idx):
            j = i
            while True:
                if j == len(detected_falls_idx):
                    break
                if detected_falls_idx[j] - detected_falls_idx[i] > cluster_win_len:
                    break
                j += 1
            clustered_falls.append(detected_falls_idx[i:j])
            i = j

        # Step 2: Match clusters to GT falls using prediction window
        matched_gt_indices = set()
        tp_cluster_indices = set()
        for cluster_idx, cluster_dets in enumerate(clustered_falls):
            for gt_idx, gt_first in enumerate(gt_falls_predict_idx_firsts):
                if gt_idx in matched_gt_indices:
                    continue
                window_start = int(gt_first - self.lower_bound)
                window_end = int(gt_first - self.prediction_gap)
                if any(window_start <= det <= window_end for det in cluster_dets):
                    tp_cluster_indices.add(cluster_idx)
                    matched_gt_indices.add(gt_idx)
                    break

        # Step 3: FP = clusters with no GT match
        falls_fp = [i for i in range(len(clustered_falls)) if i not in tp_cluster_indices]
        falls_tp = list(matched_gt_indices)

        falls_fn = [gt_falls_predict_idx_firsts[i] for i in range(len(gt_falls_predict_idx_firsts)) if i not in matched_gt_indices]

        # Sanity checks: every GT and every cluster must be fully accounted for
        assert len(tp_cluster_indices) + len(falls_fp) == len(clustered_falls),             f"Cluster accounting mismatch: {len(tp_cluster_indices)} TP + {len(falls_fp)} FP != {len(clustered_falls)} total clusters"
        assert len(falls_tp) + len(falls_fn) == len(gt_falls_predict_idx_firsts),             f"TP and FN not equal GT: {len(falls_tp)} + {len(falls_fn)} != {len(gt_falls_predict_idx_firsts)}"

        return falls_tp, falls_fp, falls_fn

    def cal_roc_anomaly(self, loss_history, centroidZ_history, gt_falls_predict_idx_firsts):
        n_gt_falls = len(gt_falls_predict_idx_firsts)
        print("How many falls?", n_gt_falls)
        tpr, fpr = [], []
        centroidZ_thr = 0.6
        for threshold in np.arange(0.0, 10.0, 0.1):
            detected_falls_idx, _ = self.detect_falls(loss_history, centroidZ_history, threshold, centroidZ_thr)
            falls_tp, falls_fp, falls_fn = self.find_tpfpfn(detected_falls_idx, gt_falls_predict_idx_firsts)
            tpr.append(len(falls_tp) / n_gt_falls)
            fpr.append(len(falls_fp))
        return tpr, fpr

    def cal_roc_centroid(self, loss_history, centroidZ_history, gt_falls_predict_idx_firsts):
        n_gt_falls = len(gt_falls_predict_idx_firsts)
        print("How many falls?", n_gt_falls)
        tpr, fpr = [], []
        anomaly_thr = 5
        for centroid_thr in np.arange(0.0, 0.6, 0.1):
            detected_falls_idx, _ = self.detect_falls(loss_history, centroidZ_history, anomaly_thr, centroid_thr)

            falls_tp, falls_fp, falls_fn = self.find_tpfpfn(detected_falls_idx, gt_falls_predict_idx_firsts)
            tpr.append(len(falls_tp) / n_gt_falls)
            fpr.append(len(falls_fp))
        return tpr, fpr

# ----------------------------------------------------------------------------

# Cell 5
if __name__ == '__main__':
    # project_path = '/Users/idoyanay/projects/personal_projects/mmfall/'
    project_path = '/content/drive/MyDrive/mmfall/'

    # # Generate the motion pattern from the original normal dataset. The proposed oversampling method is included.
    # # Uncomment if you want to regenerate the data. It may take a long time.
    # #####################################################################################################
    # print("*********************************************************************")
    # # Load the normal data file and preprocess the data
    # print("INFO: Start preprocessing the normal training dataset...")
    # train_data, test_data= data_preproc().load_bin(project_path + 'data/DS0/DS0.npy', fortrain=True)
    # # Save the normal training and testing dataset
    # np.save(project_path + 'data/normal_train_data', np.array(train_data))
    # print("INFO: The train_data is saved in " + save_datapath + 'data/normal_train_data' + ".npy")
    # np.save(project_path + 'data/normal_test_data', np.array(test_data))
    # print("INFO: The test_data is saved in " + save_datapath + 'data/normal_test_data' + ".npy")
    # #####################################################################################################

# ----------------------------------------------------------------------------

# Cell 6
    # # Uncomment if you want to retrain the models. It may take a long time. And you may expect slightly changes in the results as training progress has lot of random things.
    # # Or you can skip this section, go to next and load the saved models to check the results.
    # # Train and save the model
    # print("INFO: Load train/test data...")
    # train_data      = np.load(project_path + 'data/normal_train_data.npy', allow_pickle=True)
    # test_data       = np.load(project_path + 'data/normal_test_data.npy', allow_pickle=True)
    # print("INFO: Start HVRAE/HVRAE_SL/RAE model training and testing...")
    # model = autoencoder_mdl(model_dir = (project_path + 'saved_model/'))
    # model.HVRAE_train(train_data, test_data)
    # model.HVRAE_SL_train(train_data, test_data)
    # model.RAE_train(train_data, test_data)

    ## our edit
    # path = r'C:/Users/idosy/uni_projects/mmfall/saved_model/HVRAE_mdl.h5'

    # import os
    # print(f"File size: {os.path.getsize(path)} bytes")

    # with open(path, 'rb') as f:
    #     header = f.read(50)
    #     print(f"Header: {header}")

# ----------------------------------------------------------------------------

# Cell 7
    # Load inference dataset and the ground truth timesheet
    inferencedata_file_falls                      = project_path + 'data/DS1/DS1_4falls'
    inferencedata_falls, centroidZ_history_falls        = data_preproc().load_bin(inferencedata_file_falls + '.npy', fortrain=False)

    inferencedata_file_normal                      = project_path + 'data/DS1/DS1_4normal'
    inferencedata_normal, centroidZ_history_normal        = data_preproc().load_bin(inferencedata_file_normal + '.npy', fortrain=False)

    # ---- concatinate the falls and normal data for better initial testing ---- #
    inferencedata_falls_normal = inferencedata_falls + inferencedata_normal
    centroidZ_history_falls_normal = centroidZ_history_falls + centroidZ_history_normal

    # --- calculate the gt_falls_idx (by knowing that there are only 4 falls)
    centroidZ_history_np = np.array(centroidZ_history_falls)
    n = len(centroidZ_history_np)
    window_size = 100

    # Find the index of the minimum value in each 100-sample window
    window_mins = []
    for start in range(0, n, window_size):
        end = min(start + window_size, n)
        window = centroidZ_history_np[start:end]
        local_min_idx = np.argmin(window) + start  # offset by window start
        window_mins.append((centroidZ_history_np[local_min_idx], local_min_idx))

    # Sort by value and take the 4 smallest
    window_mins.sort(key=lambda x: x[0])
    gt_falls_idx = [idx for _, idx in window_mins[:4]]


# ----------------------------------------------------------------------------

# Cell 8

    # Load the models
    model                                   = autoencoder_mdl(model_dir = project_path + 'saved_model/')

    HVRAE_loss_history_falls                       = model.HVRAE_predict(inferencedata_falls)
    HVRAE_loss_history_normal                       = model.HVRAE_predict(inferencedata_normal)

    HVRAE_SL_loss_history_falls                    = model.HVRAE_SL_predict(inferencedata_falls)
    HVRAE_SL_loss_history_normal                    = model.HVRAE_SL_predict(inferencedata_normal)

    RAE_loss_history_falls                        = model.RAE_predict(inferencedata_falls)
    RAE_loss_history_normal                        = model.RAE_predict(inferencedata_normal)

    HVRAE_loss_history_falls_normal = HVRAE_loss_history_falls + HVRAE_loss_history_normal
    HVRAE_SL_loss_history_falls_normal = HVRAE_SL_loss_history_falls + HVRAE_SL_loss_history_normal
    RAE_loss_history_falls_normal = RAE_loss_history_falls + RAE_loss_history_normal





# ----------------------------------------------------------------------------

# Cell 9
    plt.figure()
    plt.ylim(0.0, 2.2)
    plt.xlabel("Time in Seconds")
    time_step = np.arange(0, len(centroidZ_history_normal))*0.1 # 0.1 seconds per frame
    plt.plot(time_step, centroidZ_history_normal, linewidth=2, label='Centroid Height')
    plt.plot(time_step, HVRAE_loss_history_normal, linewidth=2, label='Anomaly Level')
    plt.legend(loc="upper right")
    plt.title('HVRAE Results')
    plt.savefig(inferencedata_file_normal+'_HVRAE_prediction.png')
    plt.show()

    plt.figure()
    plt.ylim(0.0, 2.2)
    plt.xlabel("Time in Seconds")
    time_step = np.arange(0, len(centroidZ_history_normal))*0.1 # 0.1 seconds per frame
    plt.plot(time_step, centroidZ_history_normal, linewidth=2, label='Centroid Height')
    plt.plot(time_step, HVRAE_SL_loss_history_normal, linewidth=2, label='Anomaly Level')
    plt.legend(loc="upper right")
    plt.title('HVRAE_SL Results')
    plt.savefig(inferencedata_file_normal+'_HVRAE_SL_prediction.png')
    plt.show()

    plt.figure()
    plt.ylim(0.0, 2.2)
    plt.xlabel("Time in Seconds")
    time_step = np.arange(0, len(centroidZ_history_normal))*0.1 # 0.1 seconds per frame
    plt.plot(time_step, centroidZ_history_normal, linewidth=2, label='Centroid Height')
    plt.plot(time_step, RAE_loss_history_normal, linewidth=2, label='Anomaly Level')
    plt.legend(loc="upper right")
    plt.title('RAE Results')
    plt.savefig(inferencedata_file_normal+'_RAE_prediction.png')
    plt.show()

# ----------------------------------------------------------------------------

# Cell 10
    plt.figure()
    plt.ylim(0.0, 2.2)
    plt.xlabel("Time in Seconds")
    time_step = np.arange(0, len(centroidZ_history_falls))*0.1 # 0.1 seconds per frame
    plt.plot(time_step, centroidZ_history_falls, linewidth=2, label='Centroid Height')
    plt.plot(time_step, HVRAE_loss_history_falls, linewidth=2, label='Anomaly Level')
    plt.legend(loc="upper right")
    plt.title('HVRAE Results')
    plt.savefig(inferencedata_file_falls+'_HVRAE_prediction.png')
    plt.show()

    plt.figure()
    plt.ylim(0.0, 2.2)
    plt.xlabel("Time in Seconds")
    time_step = np.arange(0, len(centroidZ_history_falls))*0.1 # 0.1 seconds per frame
    plt.plot(time_step, centroidZ_history_falls, linewidth=2, label='Centroid Height')
    plt.plot(time_step, HVRAE_SL_loss_history_falls, linewidth=2, label='Anomaly Level')
    plt.legend(loc="upper right")
    plt.title('HVRAE_SL Results')
    plt.savefig(inferencedata_file_falls+'_HVRAE_SL_prediction.png')
    plt.show()

    plt.figure()
    plt.ylim(0.0, 2.2)
    plt.xlabel("Time in Seconds")
    time_step = np.arange(0, len(centroidZ_history_falls))*0.1 # 0.1 seconds per frame
    plt.plot(time_step, centroidZ_history_falls, linewidth=2, label='Centroid Height')
    plt.plot(time_step, RAE_loss_history_falls, linewidth=2, label='Anomaly Level')
    plt.legend(loc="upper right")
    plt.title('RAE Results')
    plt.savefig(inferencedata_file_falls+'_RAE_prediction.png')
    plt.show()

# ----------------------------------------------------------------------------

# Cell 11

    plt.figure()
    plt.ylim(0.0, 2.2)
    plt.xlabel("Time in Seconds")
    time_step = np.arange(0, len(centroidZ_history_falls) + len(centroidZ_history_normal))*0.1 # 0.1 seconds per frame
    plt.plot(time_step, centroidZ_history_falls_normal, linewidth=2, label='Centroid Height')
    plt.plot(time_step, HVRAE_loss_history_falls_normal, linewidth=2, label='Anomaly Level')
    plt.legend(loc="upper right")
    plt.title('HVRAE Results (falls + normal)')
    plt.savefig(inferencedata_file_falls+'_HVRAE_prediction.png')
    plt.show()

# ----------------------------------------------------------------------------

# Cell 12

    ## ---- plot the 80 to 100 frames, to try and see pattern of fall --- #
    plt.figure()
    plt.ylim(0.0, 2.2)
    plt.xlabel("Time in Seconds")
    time_step = np.arange(920, 940)*0.1 # 0.1 seconds per frame
    plt.plot(time_step, centroidZ_history_falls[910:930], linewidth=2, label='Centroid Height')
    plt.plot(time_step, HVRAE_loss_history_falls[910:930], linewidth=2, label='Anomaly Level')
    plt.legend(loc="upper right")
    plt.title('HVRAE Results')
    # plt.savefig(inferencedata_file_falls+'_HVRAE_prediction.png')
    plt.show()


# ----------------------------------------------------------------------------

# Cell 13
    # Early prediction evaluation on DS1 — skipped (no gt_falls_predict_idx_firsts for DS1)
    # print(f"gt_falls_idx: {gt_falls_idx}")
    # for gap in [1, 2, 3]:
    #     calculator = compute_metric_early_predict(prediction_gap=gap)
    #     HVRAE_ep_tpr, HVRAE_ep_fpr = calculator.cal_roc_anomaly(HVRAE_loss_history_falls_normal, centroidZ_history_falls_normal, gt_falls_predict_idx_firsts)
    #     HVRAE_SL_ep_tpr, HVRAE_SL_ep_fpr = calculator.cal_roc_anomaly(HVRAE_SL_loss_history_falls_normal, centroidZ_history_falls_normal, gt_falls_predict_idx_firsts)
    #     RAE_ep_tpr, RAE_ep_fpr = calculator.cal_roc_anomaly(RAE_loss_history_falls_normal, centroidZ_history_falls_normal, gt_falls_predict_idx_firsts)

# ----------------------------------------------------------------------------

# Cell 14
    # Load inference dataset and the ground truth timesheet
    inferencedata_file                      = project_path + 'data/DS2/DS2'
    inferencedata, centroidZ_history        = data_preproc().load_bin(inferencedata_file + '.npy', fortrain=False)
    # Ground truth time index file exists
    if os.path.exists(inferencedata_file + '.csv'):
        gt_falls_idx                        = np.genfromtxt(inferencedata_file + '.csv', delimiter=',').astype(int)
    # Maybe this file doesn't contain any falls
    else:
        gt_falls_idx                        = []

    # Load the models
    model                                   = autoencoder_mdl(model_dir = project_path + 'saved_model/')
    HVRAE_loss_history                       = model.HVRAE_predict(inferencedata)
    HVRAE_SL_loss_history                    = model.HVRAE_SL_predict(inferencedata)
    RAE_loss_history                        = model.RAE_predict(inferencedata)



# ----------------------------------------------------------------------------

# Cell 15

    # Create DS2_PREDICT: expand each GT fall to all adjacent frames with centroidZ < threshold
    centroid_thr = 0.44
    search_radius = 10
    gt_falls_predict_idx = set()
    gt_falls_predict_idx_firsts = set()
    centroidZ_np = np.array(centroidZ_history)
    n_labeled = 0
    n_skipped = 0

    for gt_frame in gt_falls_idx:
        current_fall_indexes = set()
        anchor = int(gt_frame)
        # print(f"frame index: {gt_frame}, centroid height: {centroidZ_np[anchor]:.3g}")
        print(f"gt_frame index: {anchor}")
        # If the GT frame itself isn't below threshold, search nearby for the closest one
        if centroidZ_np[anchor] > centroid_thr:
            found = False
            for offset in range(1, search_radius + 1):
                if anchor - offset >= 0 and centroidZ_np[anchor - offset] < centroid_thr:
                    anchor = anchor - offset
                    found = True
                    break
                if anchor + offset < len(centroidZ_np) and centroidZ_np[anchor + offset] < centroid_thr:
                    anchor = anchor + offset
                    found = True
                    break
            if found:
                print(f"  -> gt_frame {int(gt_frame)} centroidZ={centroidZ_np[int(gt_frame)]:.3f} > {centroid_thr}, shifted to {anchor} (offset={anchor - int(gt_frame):+d}, centroidZ={centroidZ_np[anchor]:.3f})")
            else:
                print(f"  -> gt_frame {int(gt_frame)} centroidZ={centroidZ_np[int(gt_frame)]:.3f} > {centroid_thr}, no frame below threshold within ±{search_radius} — SKIPPED")
                n_skipped += 1
                continue

        # Walk backward from anchor
        i = anchor
        while i >= 0 and centroidZ_np[i] < centroid_thr:
            current_fall_indexes.add(i)
            i -= 1
        # Walk forward from anchor
        i = anchor + 1
        while i < len(centroidZ_np) and centroidZ_np[i] < centroid_thr:
            current_fall_indexes.add(i)
            i += 1
        for index in current_fall_indexes:
            gt_falls_predict_idx.add(index)
        gt_falls_predict_idx_firsts.add(min(current_fall_indexes))
        n_labeled += 1

    gt_falls_predict_idx = np.array(sorted(gt_falls_predict_idx))
    np.savetxt(project_path + 'data/DS2/DS2_PREDICT.csv', gt_falls_predict_idx, fmt='%d', delimiter=',')
    print(f"DS2_PREDICT: {n_labeled} falls labeled, {n_skipped} skipped, {len(gt_falls_predict_idx)} total frames")
    print(f"Saved to {project_path + 'data/DS2/DS2_PREDICT.csv'}")

    gt_falls_predict_idx_firsts = np.array(sorted(gt_falls_predict_idx_firsts))
    np.savetxt(project_path + 'data/DS2/DS2_PREDICT_FIRSTS.csv', gt_falls_predict_idx_firsts, fmt='%d', delimiter=',')
    print(f"DS2_PREDICT_FIRSTS: {n_labeled} falls labeled, {n_skipped} skipped, {len(gt_falls_predict_idx_firsts)} total frames")
    print(f"Saved to {project_path + 'data/DS2/DS2_PREDICT_FIRSTS.csv'}")


# ----------------------------------------------------------------------------

# Cell 16

    # For performace evaluation
    calculator                              = compute_metric()
    HVRAE_tpr, HVRAE_fp_total                 = calculator.cal_roc(HVRAE_loss_history, centroidZ_history, gt_falls_idx)
    HVRAE_SL_tpr, HVRAE_SL_fp_total           = calculator.cal_roc(HVRAE_SL_loss_history, centroidZ_history, gt_falls_idx)
    RAE_tpr, RAE_fp_total                   = calculator.cal_roc(RAE_loss_history, centroidZ_history, gt_falls_idx)



# ----------------------------------------------------------------------------

# Cell 17

    # Plot Receiver operating characteristic (ROC) curves
    plt.figure()
    plt.xlim(-0.1, 10)
    plt.xticks(np.arange(0, 11, 1))
    plt.ylim(0.0, 1.1)
    plt.scatter(HVRAE_fp_total, HVRAE_tpr, c='r')
    plt.plot(HVRAE_fp_total, HVRAE_tpr, c='r', linewidth=2, label='HVRAE ROC')
    plt.xlim(-0.1, 10)
    plt.xticks(np.arange(0, 11, 1))
    plt.ylim(0.0, 1.1)
    plt.scatter(HVRAE_SL_fp_total, HVRAE_SL_tpr, c='b')
    plt.plot(HVRAE_SL_fp_total, HVRAE_SL_tpr, c='b', linewidth=2, label='HVRAE_SL ROC')
    plt.xlim(-0.1, 10)
    plt.xticks(np.arange(0, 11, 1))
    plt.ylim(0.0, 1.1)
    plt.scatter(RAE_fp_total, RAE_tpr, c='g')
    plt.plot(RAE_fp_total, RAE_tpr, c='g', linewidth=2, label='RAE ROC')
    plt.legend(loc="lower right")
    plt.xlabel("Number od False Alarms")
    plt.ylabel("True Fall Detection Rate")
    plt.savefig(inferencedata_file + '_ROC.png')
    plt.show()

    # plt.figure()
    # plt.xlim(-0.1,10)
    # plt.xticks(np.arange(0,11,1))
    # plt.ylim(0.0,1.1)
    # plt.scatter(HVRAE_fp_total_predict_anomaly, HVRAE_tpr_predict_anomaly, c='r')
    # plt.plot(HVRAE_fp_total_predict_anomaly, HVRAE_tpr_predict_anomaly, c='r', linewidth=2, label='HVRAE ROC (predict anomly)')
    # plt.xlim(-0.1,10)
    # plt.xticks(np.arange(0,11,1))
    # plt.ylim(0.0,1.1)
    # plt.scatter(HVRAE_fp_total_predict_centroid, HVRAE_tpr_predict_centroid, c='r')
    # plt.plot(HVRAE_fp_total_predict_centroid, HVRAE_tpr_predict_centroid, c='r', linewidth=2, label='HVRAE ROC (predict centroid)')



# ----------------------------------------------------------------------------

# Cell 18
    # Early prediction evaluation on DS2
    for gap in [1, 2, 3]:
        calculator = compute_metric_early_predict(prediction_gap=gap)
        HVRAE_ep_tpr, HVRAE_ep_fpr = calculator.cal_roc_anomaly(HVRAE_loss_history, centroidZ_history, gt_falls_predict_idx_firsts)
        HVRAE_SL_ep_tpr, HVRAE_SL_ep_fpr = calculator.cal_roc_anomaly(HVRAE_SL_loss_history, centroidZ_history, gt_falls_predict_idx_firsts)
        RAE_ep_tpr, RAE_ep_fpr = calculator.cal_roc_anomaly(RAE_loss_history, centroidZ_history, gt_falls_predict_idx_firsts)

        # plt.figure()
        # plt.xlim(-0.1, 10)
        # plt.xticks(np.arange(0, 11, 1))
        # plt.ylim(0.0, 1.1)
        # plt.scatter(HVRAE_ep_fpr, HVRAE_ep_tpr, c='r')
        # plt.plot(HVRAE_ep_fpr, HVRAE_ep_tpr, c='r', linewidth=2, label='HVRAE')
        # plt.scatter(HVRAE_SL_ep_fpr, HVRAE_SL_ep_tpr, c='b')
        # plt.plot(HVRAE_SL_ep_fpr, HVRAE_SL_ep_tpr, c='b', linewidth=2, label='HVRAE_SL')
        # plt.scatter(RAE_ep_fpr, RAE_ep_tpr, c='g')
        # plt.plot(RAE_ep_fpr, RAE_ep_tpr, c='g', linewidth=2, label='RAE')
        # plt.title(f'DS2 Early Predict ROC (anomaly sweep, gap={gap} frames)')
        # plt.xlabel('Number of False Alarms')
        # plt.ylabel('True Fall Detection Rate')
        # plt.legend(loc='lower right')
        # plt.show()
        plt.figure()
        plt.xlim(-0.1, 10)
        plt.xticks(np.arange(0, 11, 1))
        plt.ylim(0.0, 1.1)
        plt.scatter(HVRAE_ep_fpr, HVRAE_ep_tpr, c='r')
        plt.plot(HVRAE_ep_fpr, HVRAE_ep_tpr, c='r', linewidth=2, label='HVRAE')
        plt.scatter(HVRAE_SL_ep_fpr, HVRAE_SL_ep_tpr, c='b')
        plt.plot(HVRAE_SL_ep_fpr, HVRAE_SL_ep_tpr, c='b', linewidth=2, label='HVRAE_SL')
        plt.scatter(RAE_ep_fpr, RAE_ep_tpr, c='g')
        plt.plot(RAE_ep_fpr, RAE_ep_tpr, c='g', linewidth=2, label='RAE')
        plt.title(f'DS1 Early Predict ROC (anomaly sweep, gap={gap} frames)')
        plt.xlabel('Number of False Alarms')
        plt.ylabel('True Fall Detection Rate')
        plt.legend(loc='lower right')

# ----------------------------------------------------------------------------

# Cell 19
    plt.figure()
    plt.ylim(0.0, 2.2)
    plt.xlabel("Time in Seconds")
    time_step = np.arange(0, len(centroidZ_history[100:1000]))*0.1 # 0.1 seconds per frame
    plt.plot(time_step, centroidZ_history[100:1000], linewidth=2, label='Centroid Height')
    plt.plot(time_step, HVRAE_loss_history[100:1000], linewidth=2, label='Anomaly Level')
    plt.legend(loc="upper right")
    plt.title('HVRAE Results')
    plt.show()

    plt.figure()
    plt.ylim(0.0, 2.2)
    plt.xlabel("Time in Seconds")
    time_step = np.arange(0, len(centroidZ_history[100:1000]))*0.1 # 0.1 seconds per frame
    plt.plot(time_step, centroidZ_history[100:1000], linewidth=2, label='Centroid Height')
    plt.plot(time_step, HVRAE_SL_loss_history[100:1000], linewidth=2, label='Anomaly Level')
    plt.legend(loc="upper right")
    plt.title('HVRAE_SL Results')
    plt.show()

    plt.figure()
    plt.ylim(0.0, 2.2)
    plt.xlabel("Time in Seconds")
    time_step = np.arange(0, len(centroidZ_history[100:1000]))*0.1 # 0.1 seconds per frame
    plt.plot(time_step, centroidZ_history[100:1000], linewidth=2, label='Centroid Height')
    plt.plot(time_step, RAE_loss_history[100:1000], linewidth=2, label='Anomaly Level')
    plt.legend(loc="upper right")
    plt.title('RAE Results')
    plt.show()


    # Combined plot: centroidZ + all loss histories
    plt.figure(figsize=(14, 5))
    plt.ylim(0.0, 2.2)
    plt.xlabel("Time in Seconds")
    time_step = np.arange(0, len(centroidZ_history[100:1000]))*0.1
    plt.plot(time_step, centroidZ_history[100:1000], linewidth=2, label='Centroid Height', c='k')
    plt.plot(time_step, HVRAE_loss_history[100:1000], linewidth=2, label='HVRAE', c='r')
    plt.plot(time_step, HVRAE_SL_loss_history[100:1000], linewidth=2, label='HVRAE_SL', c='b')
    plt.plot(time_step, RAE_loss_history[100:1000], linewidth=2, label='RAE', c='g')
    plt.legend(loc="upper right")
    plt.title('All Models vs Centroid Height (frames 100-1000)')
    plt.show()
    def local_maxima_top_n(numbers, n=4, window=15):
        candidates = []

        for i, val in enumerate(numbers):
            lo = max(0, i - window)
            hi = min(len(numbers), i + window + 1)
            if val == max(numbers[lo:hi]):
                candidates.append((val, i))

        # Sort by value descending, take top n
        candidates.sort(reverse=True)
        return candidates[:n]

    top_four_HVRAE = local_maxima_top_n(HVRAE_loss_history, n=4, window=15)
    top_four_HVRAE_SL = local_maxima_top_n(HVRAE_SL_loss_history, n=4, window=15)
    top_four_RAE = local_maxima_top_n(RAE_loss_history, n=4, window=15)
    print("------------------")
    print(f"top four values HVRAE: {top_four_HVRAE}")
    print(f"top four values HVRAE_SL: {top_four_HVRAE_SL}")
    print(f"top four values RAE: {top_four_RAE}")
    print("------------------")
    # --- Debug: GT falls missed by HVRAE but caught by HVRAE_SL ---
    debug_threshold = 5.0
    debug_centroidZ_thr = 0.6
    debug_calculator = compute_metric_early_predict(prediction_gap=1, lower_bound=5)

    hvrae_detected, _ = debug_calculator.detect_falls(HVRAE_loss_history, centroidZ_history, debug_threshold, debug_centroidZ_thr)
    hvrae_sl_detected, _ = debug_calculator.detect_falls(HVRAE_SL_loss_history, centroidZ_history, debug_threshold, debug_centroidZ_thr)

    hvrae_tp, hvrae_fp, hvrae_fn = debug_calculator.find_tpfpfn(hvrae_detected, gt_falls_predict_idx_firsts)
    hvrae_sl_tp, hvrae_sl_fp, hvrae_sl_fn = debug_calculator.find_tpfpfn(hvrae_sl_detected, gt_falls_predict_idx_firsts)

    print(f"HVRAE:    TP={len(hvrae_tp)}, FP={len(hvrae_fp)}, FN={len(hvrae_fn)}")
    print(f"HVRAE_SL: TP={len(hvrae_sl_tp)}, FP={len(hvrae_sl_fp)}, FN={len(hvrae_sl_fn)}")

    # GT falls missed by HVRAE but caught by HVRAE_SL
    hvrae_fn_set = set(hvrae_fn)
    hvrae_sl_tp_gt_firsts = set(gt_falls_predict_idx_firsts[i] for i in hvrae_sl_tp)
    missed_by_hvrae_caught_by_sl = sorted(hvrae_fn_set & hvrae_sl_tp_gt_firsts)

    print(f"\nGT falls missed by HVRAE but caught by HVRAE_SL: {len(missed_by_hvrae_caught_by_sl)}")
    print(f"Frame indices: {missed_by_hvrae_caught_by_sl}")

    for gt_first in missed_by_hvrae_caught_by_sl:
        # Print debug info
        z_at_gt = centroidZ_history[gt_first]
        win_start = max(0, gt_first - 9)
        z_drop = centroidZ_history[win_start] - centroidZ_history[gt_first]
        print(f"\n--- GT fall at frame {gt_first} ---")
        print(f"  centroidZ at gt_first: {z_at_gt:.3f}")
        print(f"  centroidZ drop (10-frame window): {z_drop:.3f}")

        lo = max(0, gt_first - 15)
        hi = min(len(HVRAE_loss_history), gt_first + 16)
        print(f"  HVRAE    anomaly [{lo}:{hi}]: {[f'{v:.2f}' for v in HVRAE_loss_history[lo:hi]]}")
        print(f"  HVRAE_SL anomaly [{lo}:{hi}]: {[f'{v:.2f}' for v in HVRAE_SL_loss_history[lo:hi]]}")

        # Plot local view
        plot_lo = max(0, gt_first - 30)
        plot_hi = min(len(HVRAE_loss_history), gt_first + 11)
        frames = np.arange(plot_lo, plot_hi)

        fig, ax1 = plt.subplots(figsize=(12, 4))
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Centroid Z (m)', color='k')
        ax1.plot(frames, centroidZ_history[plot_lo:plot_hi], 'k-', linewidth=2, label='centroidZ')
        ax1.tick_params(axis='y', labelcolor='k')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Anomaly Score', color='gray')
        ax2.plot(frames, HVRAE_loss_history[plot_lo:plot_hi], 'r-', linewidth=1.5, label='HVRAE')
        ax2.plot(frames, HVRAE_SL_loss_history[plot_lo:plot_hi], 'b-', linewidth=1.5, label='HVRAE_SL')
        ax2.axhline(y=debug_threshold, color='gray', linestyle='--', alpha=0.5, label=f'threshold={debug_threshold}')
        ax2.tick_params(axis='y', labelcolor='gray')

        ax1.axvline(x=gt_first, color='green', linestyle='--', linewidth=2, label=f'GT first={gt_first}')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize='small')
        plt.title(f'GT fall at frame {gt_first}: missed by HVRAE, caught by HVRAE_SL')
        plt.tight_layout()
        plt.show()