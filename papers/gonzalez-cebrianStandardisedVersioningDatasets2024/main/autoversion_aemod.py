"""
Script Name: autoversion_aemod.py
Description: this module provides the functionality to apply versioning using 
autoencoders (AE) to compute the data drift term. To do so, it includes the necessary 
functions to perform model training and compute the version attributes.

Author:
- Name: Alba González-Cebrián, Fanny Rivera-Ortiz, Jorge Mario Cortés-Mendoza, Adriana E. Chis, Michael Bradford, Horacio González-Vélez
- Email: Alba.Gonzalez-Cebrian@ncirl.ie

License: MIT License
- License URL: https://opensource.org/license/mit/
"""
# Rest of your code goes here

# Module with versioning applying autoencoders
import copy
import datetime
import os
import pandas as pd
# FRO: Added the following line to remove the SettingWithCopyWarning warnings
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import random
import tensorflow as tf
# import tensorflow.keras as keras
# FRO: Modified the previous line to load the keras library in a different way
from tensorflow import keras
import keras_tuner as kt
import matplotlib.pyplot as plt
import autoversion_gen as av_gen 
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from keras import layers
from keras.models import Model
from sklearn.preprocessing import StandardScaler
os.environ['TF_DETERMINISTIC_OPS'] = '1'

def versionPS(df, path_logs, path_tuner, modeclean = "r&c", nanvalue = np.nan, nrep = 50, alpha = 0.05, outdict = False, loadfile = True, 
              hname = None, usecols = None, skiprows=0, dtype = float, dopickle = False, delimiter = None, respath = "", decimal = ".", dojson = False, dstitle = "",
              epochs_tuner=40, epochs_fit=20, num_plots = 5, verbose=0,  mse_model_type = "splines", tr_val_split = 0.8, steps_list = [5, 10, 15, 30, 50], 
              noisefactor = 0.05, figs_path = "", resultsfile = "", cat_threshold = 10, silent = False, silentfig = True):
    """
    Apply versioning based on autoencoders to the dataset.

    Args:
        df (pd.DataFrame): The input dataset.
        path_logs (str): The path to store the logs.
        path_tuner (str): The path for tuner.
        modeclean (str, optional): The mode of cleaning. Default is "r&c".
        nanvalue (float, optional): The value representing missing data. Default is np.nan.
        drift_dist (str, optional): The drift distance measure. Default is "dE".
        nrep (int, optional): The number of repetitions. Default is 50.
        alpha (float, optional): The alpha value. Default is 0.05.
        outdict (bool, optional): Flag to output dictionary. Default is False.
        loadfile (bool, optional): Flag to load file. Default is True.
        hname (str, optional): The name for hypermodel. Default is None.
        usecols (list, optional): The list of columns to use. Default is None.
        skiprows (int, optional): The number of rows to skip. Default is 0.
        dtype (type, optional): The data type. Default is float.
        dopickle (bool, optional): Flag to pickle the data. Default is False.
        delimiter (str, optional): The delimiter. Default is None.
        respath (str, optional): The path for results. Default is "".
        decimal (str, optional): The decimal representation. Default is ".".
        dojson (bool, optional): Flag to use JSON format. Default is False.
        dstitle (str, optional): The title for the dataset. Default is "".
        epochs_tuner (int, optional): The number of epochs for tuning. Default is 40.
        epochs_fit (int, optional): The number of epochs for fitting. Default is 20.
        num_plots (int, optional): The number of plots. Default is 5.
        verbose (int, optional): The verbosity level. Default is 0.
        mse_model_type (str, optional): The type of MSE model. Default is "splines".
        tr_val_split (float, optional): The train/validation split ratio. Default is 0.8.
        steps_list (int, optional): The list with the number of time steps. Default is [5, 10, 15, 30, 50].
        noisefactor (float, optional): The noise factor. Default is 0.05.
        figs_path (str, optional): The path for saving figures. Default is "".
        resultsfile (str, optional): The path for saving results. Default is "".
        cat_threshold (int, optional): integer indicating the maximum number of categories for a variable to be considered categorical. Deafaults to 10.
        silent (bool, optional): Boolean value indicating if comments are printed in the screen or not. Defaults to True.
        silentfig (bool, optional): Boolean value indicating if figures are printed in the screen during the execution or not. Defaults to True.

    Returns:
        dict or None: The version information if `outdict` is True, else None.
    """
    # Set random seeds for libraries
    #np.random.seed(random_seed)
    #tf.random.set_seed(42)
    #random.seed(random_seed)

    # Load dataset
    if loadfile:
        df = av_gen.load_df(copy.deepcopy(df), hname = hname, usecols = usecols, skiprows=skiprows, dtype = dtype, dopickle = dopickle, delimiter = delimiter, 
                            decimal = decimal, dojson = dojson)  
    
    # Clean it (deal with empty rows, columns, columns with null variance, etc.)
    dfps_clean = av_gen.clean_ds(df, mode = modeclean, nanvalue = nanvalue, threshold = cat_threshold, onehot=False, logcounts=False)

    # Fit versioning models of PS
    tf.get_logger().setLevel('ERROR') 
    PSv = version_info_ae(dfps_clean["dfclean"], path_logs, path_tuner, mode = "ps", alpha = alpha, nrep = nrep, dstitle = dstitle,
                          epochs_tuner=epochs_tuner, epochs_fit=epochs_fit, num_plots = num_plots,  mse_model_type = mse_model_type, 
                          tr_val_split = tr_val_split, steps_list = steps_list, noisefactor = noisefactor, figs_path = figs_path,
                          resultsfile = resultsfile, silent = silent, silentfig = silentfig)
    PSv["ohe"] = dfps_clean["ohe"]
    PSv["vbles_in_prepro"] = dfps_clean["vbles_in_prepro"]
    tf.keras.backend.clear_session()
    # Store the PS information 
    return(PSv)


def versionDER(df, ps_version_inf, path_logs, path_tuner , modeclean = "r&c", nanvalue = np.nan, drift_dist = "dE", outdict = False, loadfile = True, hname = None,
               usecols = None, skiprows=0, dtype = float, dopickle = False, delimiter = None, respath = "", decimal = ".", dojson = False, dstitle = "",
               epochs_tuner=40, epochs_fit=20, num_plots = 5, cat_threshold = 10, silent = False, silentfig=True):
    """
    Apply versioning based on autoencoders to the dataset with respect to the reference version.

    Args:
        df (pd.DataFrame): The input dataset.
        ps_version_inf (dict or str): The reference version information or path to the reference version information.
        path_logs (str): The path to store the logs.
        path_tuner (str): The path for tuner.
        modeclean (str, optional): The mode of cleaning. Default is "r&c".
        nanvalue (float, optional): The value representing missing data. Default is np.nan.
        drift_dist (str, optional): The drift distance measure. Default is "dE".
        outdict (bool, optional): Flag to output dictionary. Default is False.
        loadfile (bool, optional): Flag to load file. Default is True.
        hname (str, optional): The name for hypermodel. Default is None.
        usecols (list, optional): The list of columns to use. Default is None.
        skiprows (int, optional): The number of rows to skip. Default is 0.
        dtype (type, optional): The data type. Default is float.
        dopickle (bool, optional): Flag to pickle the data. Default is False.
        delimiter (str, optional): The delimiter. Default is None.
        respath (str, optional): The path for results. Default is "".
        decimal (str, optional): The decimal representation. Default is ".".
        dojson (bool, optional): Flag to use JSON format. Default is False.
        dstitle (str, optional): The title for the dataset. Default is "".
        epochs_tuner (int, optional): The number of epochs for tuning. Default is 40.
        epochs_fit (int, optional): The number of epochs for fitting. Default is 20.
        num_plots (int, optional): The number of plots. Default is 5.
        cat_threshold (int, optional): integer indicating the maximum number of categories for a variable to be considered categorical. Deafaults to 10.
        silent (bool, optional): Boolean value indicating if comments are printed in the screen or not. Defaults to True.
        silentfig (bool, optional): Boolean value indicating if figures are printed in the screen during the execution or not. Defaults to True.

    Returns:
        dict or None: The version information if `outdict` is True, else None.
    """

    # Load dataset
    if loadfile:
        df = av_gen.load_df(copy.deepcopy(df), hname = hname, usecols = usecols, skiprows=skiprows, dtype = dtype, dopickle = dopickle, delimiter = delimiter, 
                            respath = respath, decimal = decimal, dojson = dojson)
    if not isinstance(ps_version_inf, dict): PSv = av_gen.json2dict(ps_version_inf, model_type = "ae")
    else: PSv = ps_version_inf.copy()

    # Clean it (deal with empty rows, columns, columns with null variance, etc.)
    dfps_clean_new = av_gen.clean_ds(df, mode="impute", delete_null_var=False, onehot=False, logcounts=False, ohe=ps_version_inf["ohe"], threshold = cat_threshold,
                                     fixedvariables = list(PSv["vbles_in"]))["dfclean"]
    
    # Compute data drift according to models from PS
    tf.get_logger().setLevel('ERROR') 
    DERv = version_info_ae(dfps_clean_new, path_logs, path_tuner, mode = "der", PSv = PSv, dstitle = dstitle, epochs_tuner=epochs_tuner, 
                           epochs_fit=epochs_fit, num_plots = num_plots, silent = silent, silentfig = silentfig)
    tf.keras.backend.clear_session()

    # Store the new version attributes 
    return(DERv)


def version_info_ae(X, path_logs, path_tuner, mode = "ps", PSv = None, nrep = 30, alpha = 0.05, dstitle = "", epochs_tuner = 40, epochs_fit = 20, 
                    num_plots = 5, mse_model_type = "reg", tr_val_split = 0.8, steps_list = [5, 10, 15, 30, 50], noisefactor = 0.05, figs_path = "", 
                    resultsfile = "", silent = False, silentfig = True):
    """
    Obtain version information based on autoencoders.

    Args:
        X (pd.DataFrame): The input dataset.
        path_logs (str): The path to store the logs.
        path_tuner (str): The path for tuner.
        mode (str, optional): The mode of versioning. Default is "ps".
        PSv (dict or None, optional): The reference version information or None. Default is None.
        drift_dist (str, optional): The drift distance measure. Default is "S".
        nrep (int, optional): The number of repetitions. Default is 30.
        alpha (float, optional): The alpha value. Default is 0.05.
        dstitle (str, optional): The title for the dataset. Default is "".
        epochs_tuner (int, optional): The number of epochs for tuning. Default is 40.
        epochs_fit (int, optional): The number of epochs for fitting. Default is 20.
        num_plots (int, optional): The number of plots. Default is 5.
        mse_model_type (str, optional): The type of MSE model. Default is "reg".
        tr_val_split (float, optional): The train/validation split ratio. Default is 0.8.
        steps_list (int, optional): The list with the number of time steps. Default is [5, 10, 15, 30, 50].
        noisefactor (float, optional): The noise factor. Default is 0.05.
        figs_path (str, optional): The path for saving figures. Default is "".
        resultsfile (str, optional): The path for saving results. Default is "".
        silent (bool, optional): Boolean value indicating if comments are printed in the screen or not. Defaults to True.
        silentfig (bool, optional): Boolean value indicating if figures are printed in the screen during the execution or not. Defaults to True.

    Returns:
        dict: The version information.
    """

    if mode == "ps":
        X_train, X_test = train_test_split(X, train_size = tr_val_split, shuffle = False, random_state=42)        
        # Apply k-fold (l.o.o) to obtain C.I. on the distance parameters (for each component)
        PSinfo = vparams_ae(X_train, path_logs, path_tuner, dstitle = dstitle, epochs_tuner = epochs_tuner, epochs_fit = epochs_fit, num_plots = num_plots, 
                            steps_list = steps_list, noisetr = noisefactor==0, noisefactor = noisefactor, figs_path = figs_path, resultsfile=resultsfile,
                            silent = silent, silentfig = silentfig)
        print("Data model -  DONE")

        PSinfo["mse_model"] = av_gen.fit_mse_model(X_train, PSinfo, vparams_ae, path_logs = path_logs, path_tuner = path_tuner, dstitle = dstitle, 
                                                   mse_model_type = mse_model_type, figs_path = figs_path, resultsfile = resultsfile, silent = silent, silentfig = silentfig)
        print("Permutation test -  DONE")

        xtest_sc = pd.DataFrame(PSinfo["scaler"].transform(X_test[PSinfo["vbles_in"]]), columns = PSinfo["vbles_in"])
        # Obtain the code
        xtest_rec = pd.DataFrame(PSinfo["model"].decoder(PSinfo["model"].encoder(tf.convert_to_tensor(xtest_sc)).numpy()), columns = PSinfo["vbles_in"])     

        tf.keras.backend.clear_session()
        PSinfo["R2"] = r2_score(xtest_sc.stack(), xtest_rec.stack()) if r2_score(xtest_sc.stack(), xtest_rec.stack())>0 else 0
        PSinfo["data_model"] = X.dtypes
        PSinfo["timestamp"] = datetime.datetime.now(tz=datetime.timezone.utc)
        PSinfo["vtag"] = "1.0" + ".<" + PSinfo["timestamp"].strftime("%m/%d/%Y - %H:%M:%S") + ">"
        PSinfo["n"] = len(X) - 1
        return(PSinfo)
    
    if mode == "der":
        dfold = X.select_dtypes(include=np.number)
        if np.shape(dfold)[1] == 1:
            dfnum_steps, _ = av_gen.create_dataset(dfold, len(PSv["vbles_in"]))
            df = pd.DataFrame(dfnum_steps, columns = PSv["vbles_in"])
        else: df = dfold
        # Obtain distance and check with the intervals
        
        Dinfo = vparams_ae(X, path_logs, path_tuner, ps_aemodel = PSv, dstitle = dstitle, epochs_tuner = epochs_tuner, epochs_fit = epochs_fit, 
                           num_plots = num_plots, silent = silent, silentfig = silentfig)
        dE = av_gen.e_distance(Dinfo['mse'], PSv, vparams_ae, PSv["mse_model"], path_logs = path_logs, path_tuner = path_tuner)[0]
        v_major_ps, v_minor_ps, v_patch_ps = PSv["vtag"].split(".")
        
        # Get the new values for each one of the version terms:
        # 1) Order of the calculated distance
        
        Dinfo["data_model"] = df.dtypes
        Dinfo["data_drift"] = int(dE) if dE <= 100 else int(100)
        Dinfo["timestamp"] = datetime.datetime.now(tz=datetime.timezone.utc)
        Dinfo["vtag"] = str(int(v_major_ps) + int(not PSv["data_model"].equals(df.dtypes))) + "." + str(int(dE if dE <= 100 else 100)) + ".<" + Dinfo["timestamp"].strftime("%m/%d/%Y - %H:%M:%S") + ">"
        return(Dinfo)
 
class MyAE(Model):
    """
    Custom Autoencoder model based on Keras's Model class.
    """
    def __init__(self, hp, numPredictors):
        super().__init__()
        self.encoder = tf.keras.Sequential()
        self.decoder = tf.keras.Sequential()
        collectLayers = []
        for i in range(hp.Int("num_layers", 1, 3)):
            collectLayers.append(layers.Dense(units = hp.Int(f"units_{i}", min_value = min([2, numPredictors]), max_value = numPredictors, step = 2),
                                              activation = hp.Choice("activation", ["relu", "tanh"])))
            
        # Build Encoder
        for l in collectLayers:
            #if hp.Boolean("batch_normalization"): self.encoder.add(layers.BatchNormalization())
            self.encoder.add(l)
        
        # Build Decoder
        if len(collectLayers) >= 2:
            for l in collectLayers[len(collectLayers)-2::-1]:
                #if hp.Boolean("batch_normalization"): self.decoder.add(layers.BatchNormalization())
                self.decoder.add(copy.deepcopy(l))
        #else:
            #if hp.Boolean("batch_normalization"): self.decoder.add(layers.BatchNormalization())
        self.decoder.add(layers.Dense(numPredictors, activation="linear"))
       
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
class MyAEHyperModel(kt.HyperModel):
    """
    HyperModel class for Autoencoder.
    """    
    def __init__(self, numPredictors): self.numPredictors = numPredictors
    def build(self, hp):
        model = Getautoencoder(hp, self.numPredictors)
        return model
     
def Getautoencoder(hp, numPredictors):
    """
    Build an Autoencoder model.

    Args:
        hp: Hyperparameters.
        numPredictors (int): Number of predictors.

    Returns:
        tf.keras.Model: The Autoencoder model.
    """
    autoencoder = MyAE(hp, numPredictors)
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    loss = hp.Choice("loss", ["mse", "mae", "huber_loss"])       
    autoencoder.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate), loss=loss, 
                        metrics=[keras.metrics.MeanAbsoluteError(), keras.metrics.MeanSquaredError()])
    # Save a checkpoint
    checkpoint = tf.train.Checkpoint(model=autoencoder)
    checkpoint.save("/tmp/ckpt")
    return autoencoder

def vparams_ae(X, path_logs, path_tuner, ps_aemodel = None, noisetr = True, dstitle="", epochs_tuner = 40, epochs_fit = 20, num_plots = 5, tr_val_split = 0.8,
               noisefactor = 0.5, steps_list = [5, 10, 15, 30, 50], figs_path = "", resultsfile = "", silent = False, silentfig = True):
    """
    Obtain parameters based on autoencoders.

    Args:
        X (pd.DataFrame): The input dataset.
        path_logs (str): The path to store the logs.
        path_tuner (str): The path for tuner.
        ps_aemodel (dict or None, optional): The reference version information or None. Default is None.
        noisetr (bool, optional): Flag to add noise to the training data. Default is True.
        dstitle (str, optional): The title for the dataset. Default is "".
        epochs_tuner (int, optional): The number of epochs for tuning. Default is 40.
        epochs_fit (int, optional): The number of epochs for fitting. Default is 20.
        num_plots (int, optional): The number of plots. Default is 5.
        tr_val_split (float, optional): The train/validation split ratio. Default is 0.8.
        noisefactor (float, optional): The noise factor. Default is 0.5.
        steps_list (int, optional): The list with the numbers of time steps. Default is [5, 10, 15, 30, 50].
        figs_path (str, optional): The path for saving figures. Default is "".
        resultsfile (str, optional): The path for saving results. Default is "".
        silent (bool, optional): Boolean value indicating if comments are printed in the screen or not. Defaults to True.
        silentfig (bool, optional): Boolean value indicating if figures are printed in the screen during the execution or not. Defaults to True.

    Returns:
        dict: The version information.
    """

    # Select only numeric variables
    # Xnum_df = X.select_dtypes(include=np.number)
    Xnum_df = X
    numPredictors = np.shape(Xnum_df)[1]
    #tf.random.set_seed(42)
    if ps_aemodel is None:
        train_data, test_data = train_test_split(Xnum_df, train_size = tr_val_split, shuffle = False, random_state=42)

        # Scale the data
        xscaler = StandardScaler()
        xscaler.fit(train_data)
        train_data_sc = pd.DataFrame(xscaler.transform(train_data), columns = train_data.columns.values)
        test_data_sc = pd.DataFrame(xscaler.transform(test_data[train_data.columns]), columns = train_data.columns.values)
        
        # Generation of noisy artificial data 
        if noisetr:
            train_noisy_dataf, train_rep_dataf = av_gen.make_noisy_data(train_data_sc, noise_factor = noisefactor, dstitle = dstitle, num_plots = num_plots, 
                                                                        figs_path = figs_path + resultsfile, silentfig = silentfig)
        else:
            train_noisy_dataf, train_rep_dataf = av_gen.make_noisy_data(train_data_sc, noise_factor = 0, dstitle = dstitle, num_plots = num_plots, 
                                                                        figs_path = figs_path + resultsfile, silentfig = silentfig)

        if numPredictors > 1:
            # Hyper-parameters search 
            tf.keras.backend.clear_session()
            keras.utils.set_random_seed(42)
            tf.random.set_seed(42)
            #tf.get_logger().setLevel('ERROR') 
            tuner = kt.RandomSearch(hypermodel=MyAEHyperModel(numPredictors), objective="val_mean_squared_error", max_trials=30, 
                                    executions_per_trial=1, overwrite=True, directory=path_tuner, project_name="testNNps", seed = 42)
            tuner.search(train_noisy_dataf.to_numpy(), train_rep_dataf.to_numpy(), epochs=epochs_tuner, validation_split=0.2, verbose=0)
            models = tuner.get_best_models(num_models=1)
            
            # Fit all the candiates
            for m in models:
                m.build(input_shape=(None,np.shape(train_noisy_dataf)[1]))
                m.summary()
            
            #cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path_logs, save_weights_only=True, verbose=1)
            #callbacks = [cp_callback]
            # Get the best model for that number of steps and store it
            best_hp = tuner.get_best_hyperparameters()[0]
            model = Getautoencoder(best_hp, numPredictors)
            ae_train = model.fit(train_noisy_dataf, train_rep_dataf, epochs=epochs_fit, batch_size=32, shuffle=False, verbose = 0, 
                                 use_multiprocessing = True, validation_data=(test_data_sc.to_numpy(), test_data_sc.to_numpy()))
            
            try:
                plt.figure()
                plt.plot(range(20), ae_train.history['loss'], label='Training loss')
                plt.plot(range(20), ae_train.history['val_loss'], label='Validation loss')
                plt.title('Model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend()
                plt.savefig(figs_path + resultsfile +  dstitle + 'ae-loss.png', dpi = 300)
                if not silentfig:
                    plt.show()
                    plt.close()
            except Exception as e: 
                print(e)  
                
            # Obtain the MSE for the validation data set 
            print("Best model - DONE")
            decoded_data = pd.DataFrame(model.predict(test_data_sc), columns = test_data_sc.columns)
            r2score = np.round(r2_score(test_data_sc.stack(), decoded_data.stack()), 4)
            rec_error = test_data_sc.to_numpy() - decoded_data.to_numpy()
            if r2score<0:
                r2score = 1 - np.sum(np.square(rec_error))/np.sum(np.square(test_data_sc.to_numpy()))
            rss = np.sum(np.square(rec_error))
            mse = np.mean(np.square(rec_error))

            tf.keras.backend.clear_session()
            ps_version_meta = {"model" : model, "scaler" : xscaler, "vbles_in": Xnum_df.columns.values, "rss": rss, "mse": mse, "n": np.shape(Xnum_df)[0],
                               "model_class" : "ae", "R2_ps_opt": r2score}
            
        else:
            # Split data into train and test sets
            mselist = []
            modelslist = []
            metaversionlist = []
            stepslist = [x for x in steps_list if x < np.shape(train_data)[0] and x < np.shape(test_data)[0]]
            tf.keras.backend.clear_session()
            keras.utils.set_random_seed(42)
            tf.random.set_seed(42)
            for numSteps in stepslist:
                train_noisy_data_steps, __ = av_gen.create_dataset(train_noisy_dataf, numSteps)
                train_rep_data_steps, __ = av_gen.create_dataset(train_rep_dataf, numSteps)
                # The data is not scaled --> Scale it            
                test_data_steps, __ = av_gen.create_dataset(test_data_sc, numSteps)
                # Hyper-parameters search 
                
                #tf.get_logger().setLevel('ERROR') 
                tuner = kt.RandomSearch(hypermodel=MyAEHyperModel(numSteps), objective="val_mean_squared_error", max_trials=30, 
                                        executions_per_trial=1, overwrite=True, directory=path_tuner, 
                                        project_name="testNNps", seed = 42)
                
                tuner.search(train_noisy_data_steps.to_numpy(), train_rep_data_steps.to_numpy(), epochs = epochs_tuner,
                             validation_split=0.2, verbose=0)
                
                models = tuner.get_best_models(num_models=1)
                # Fit all the candiates
                for m in models:
                    m.build(input_shape=(None,np.shape(train_noisy_data_steps)[1]))
                    m.summary()
                tuner.results_summary()
                
                cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path_logs, save_weights_only=True, verbose=1)
                best_hp = tuner.get_best_hyperparameters()[0]
                model = Getautoencoder(best_hp, numPredictors)
                history = model.fit(train_noisy_data_steps, train_rep_data_steps, epochs = epochs_fit, batch_size=32, shuffle=True, verbose = 1, use_multiprocessing = True,
                                    callbacks = [cp_callback], validation_data=(test_data_steps.to_numpy(), test_data_steps.to_numpy()))
                
                try:
                    plt.figure()
                    plt.plot(range(20), history['loss'], label='Training loss')
                    plt.plot(range(20), history['val_loss'], label='Validation loss')
                    plt.title('Model loss' + str(numSteps) + " time steps" )
                    plt.ylabel('loss')
                    plt.xlabel('epoch')
                    plt.legend()
                    plt.savefig(figs_path + dstitle + 'ae-loss.png', dpi = 300)
                    if not silentfig:
                        plt.show()
                        plt.close()
                except Exception as e: 
                    print(e)  
                    
                # Obtain the code
                encoded_dataval = model.encoder(test_data_steps.to_numpy())
                decoded_data = model.decoder(encoded_dataval)
                r2score = r2_score(test_data_steps, decoded_data)
                rec_error = test_data_steps.to_numpy() - decoded_data
                if r2score<0:
                    r2score = 1 - np.sum(np.square(rec_error))/np.sum(np.square(test_data_steps))
                rss = np.sum(np.square(rec_error), axis=None)
                mse = np.mean(np.square(rec_error), axis=None)
                # Get best model
                modelslist.append(model)
                mselist.append(mse)
                            
                ps_version_meta = {"model" : model, "scaler" : xscaler, "vbles_in": Xnum_df.columns.values, "R2_ps_opt": r2score, "rss": rss,
                                   "mse": mse, "n": np.shape(train_noisy_data_steps)[0], "nsteps": np.shape(train_noisy_data_steps)[1],
                                   "step_names": train_noisy_data_steps.columns.values, "model_class" : "ae"}
                metaversionlist.append(ps_version_meta)
  
            tf.keras.backend.clear_session()
            ps_version_meta = metaversionlist[int(min(np.where(mselist == min(mselist))[0]))]
    
    else:
        
        model = ps_aemodel["model"]
        xscaler = ps_aemodel["scaler"]

        rem_vars = list(set(Xnum_df.columns.values) - set(ps_aemodel["vbles_in"]))
        X_new = Xnum_df.drop(columns = rem_vars, index = Xnum_df.index.values)
        if np.all([x in X_new.columns for x in ps_aemodel["vbles_in"]]):
            vbles_der_in = ps_aemodel["vbles_in"]
        else:
            vbles_der_in = ps_aemodel["vbles_in"][list(x in X_new.columns for x in ps_aemodel["vbles_in"])]

        X_vblesps = pd.DataFrame(np.zeros((np.shape(Xnum_df)[0], len(vbles_der_in))), columns =vbles_der_in, index=Xnum_df.index.values)
        X_vblesps[vbles_der_in] = Xnum_df[vbles_der_in]
        X_sc = pd.DataFrame(xscaler.transform(X_vblesps), columns = X_vblesps.columns.values, index = X_vblesps.index.values)

        if len(ps_aemodel["vbles_in"]) == 1:
            X_sc, __ = av_gen.create_dataset(X_sc, ps_aemodel["nsteps"])
        
        # Obtain the code
        encoded_data = model.encoder(X_sc.to_numpy())
        decoded_data = pd.DataFrame(model.decoder(encoded_data), index = X_sc.index, columns = X_sc.columns)
        r2score = r2_score(X_sc.stack(), decoded_data.stack())
        rec_error = X_sc - decoded_data
        rss = np.sum(np.square((rec_error.stack())))
        mse = np.mean(np.square((rec_error.stack())))
        tf.keras.backend.clear_session()
        ps_version_meta = {"model" : model, "scaler" : xscaler, "mse": mse, "n": np.shape(X_sc)[0], "rss": rss, "model_class" : "ae", "R2_ps_opt": r2score}          
    return(ps_version_meta)

def filter_commonvars(ref_list, X):
    """
    Filter the dataset to include only common variables present in the reference list.

    Args:
        ref_list (list): The reference list of variables.
        X (pd.DataFrame): The input dataset.

    Returns:
        pd.DataFrame: The filtered dataset.
        list: The removed variables.
    """

    rem_vars = list(set(X.columns.values) - set(ref_list))

    X_new = X.drop(columns = rem_vars, index = X.index.values)

    return X_new, rem_vars
