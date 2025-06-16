"""
Script Name: sim_experiments.py
Description: this module provides the set of functions to assist in a variety of tasks 
related to the experimentation with the time series datasets. This toolkit enables you 
to handle tasks like exporting performance metrics to Excel files, executing Primary 
Source (PS) versioning, initializing dictionaries for storing the versioning results
across multiple repetitions, comparing and filtering variables across datasets, 
calculating versioning metrics, and more. In addition, you can use this module to modify 
your data emulating the creation (C), update (U) and deletion (D) experiments. 
Furthermore, it can help you extract seasonal and trend components from time series 
data and visualize these components. In a technical sense, sim_experiments.py serves 
for the execution of a the CUD experiments and analyses related to time series data.

Author:
- Name: Alba González-Cebrián, Fanny Rivera-Ortiz, Jorge Mario Cortés-Mendoza, Adriana E. Chis, Michael Bradford, Horacio González-Vélez
- Email: Alba.Gonzalez-Cebrian@ncirl.ie

License: MIT License
- License URL: https://opensource.org/license/mit/
"""
import numpy as np
import pandas as pd
import pickle
import time
import sys
import os
import random
import re 
import autoversion_service as av
import autoversion_aemod as avae
import autoversion_gen as av_gen
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import KFold
from matplotlib.lines import Line2D
os.environ['PYTHONHASHSEED'] = '42'
plt.rcParams['figure.max_open_warning'] = 10 

def exp_write_results(perf_metrics, path_resul, do_pca=True, do_ae=True):
    """
    Writes the performance metrics to an Excel file.

    Args:
        perf_metrics (dict): Dictionary containing the performance metrics.
        path_resul (str): Path to the results file.
        do_pca (bool, optional): Whether to include PCA metrics. Defaults to True.
        do_ae (bool, optional): Whether to include AE metrics. Defaults to True.
        do_ens (bool, optional): Whether to include ensemble metrics. Defaults to True.
    """
    # Prepare the data to be written
    if do_pca:
        pca_dict = {
            "dP": {"drift": pd.DataFrame.from_dict(perf_metrics["dP"]), "version": pd.DataFrame.from_dict(perf_metrics["vP"]),
                   "time": pd.DataFrame.from_dict(perf_metrics["tP"])},
            "dE": {"drift": pd.DataFrame.from_dict(perf_metrics["dEPCA"]), "version": pd.DataFrame.from_dict(perf_metrics["vEPCA"]),
                   "time": pd.DataFrame.from_dict(perf_metrics["tEPCA"])}
        }
    if do_ae:
        ae_dict = {
            "drift": pd.DataFrame.from_dict(perf_metrics["dEAE"]), "version": pd.DataFrame.from_dict(perf_metrics["vEAE"]),
            "time": pd.DataFrame.from_dict(perf_metrics["tEAE"])
        }

    # Write to multiple sheets in an Excel file
    with pd.ExcelWriter(path_resul + '.xlsx') as writer:
        if do_pca:
            for k in pca_dict["dP"].keys():
                pca_dict["dP"][k].to_excel(writer, sheet_name=k + "-dP")
            for k in pca_dict["dE"].keys():
                pca_dict["dE"][k].to_excel(writer, sheet_name=k + "-dE_PCA")
        if do_ae:
            for k in ae_dict.keys():
                ae_dict[k].to_excel(writer, sheet_name=k + "-dE_AE")


def exp_PSinfo(dataPS, PS_dic, v_time, resultspath="", mod_pca=True, mod_ae=False, naval=np.nan, dstitle="", mse_model_type="splines",
               tr_val_split=0.8, noisefactor=0.05, yname=None, resultsfile="", cat_threshold=10, silent = True, silentfig = True):
    """
    Performs the Primary Source (PS) versioning and records the performance metrics.

    Args:
        dataPS (pd.DataFrame): Primary Source data.
        PS_dic (dict): Dictionary to store the PS versions and metrics.
        v_time (dict): Dictionary to store the versioning time for each metric.
        resultspath (str, optional): Path to the results directory. Defaults to "".
        mod_pca (bool, optional): Whether to perform PCA versioning. Defaults to True.
        mod_ae (bool, optional): Whether to perform AE versioning. Defaults to False.
        mod_ens (bool, optional): Whether to perform ensemble versioning. Defaults to True.
        naval (float, optional): NaN value to use for imputation. Defaults to np.nan.
        dstitle (str, optional): Title for the dataset. Defaults to "".
        mse_model_type (str, optional): MSE model type. Defaults to "splines".
        tr_val_split (float, optional): Train/Validation split ratio. Defaults to 0.8.
        noisefactor (float, optional): Noise factor for AE versioning. Defaults to 0.05.
        yname (str, optional): Name of the target variable. Defaults to None.
        resultsfile (str, optional): Name of the results file. Defaults to "".
    """
    try:
        if isinstance(dataPS, list):
            if len(dataPS) > 0:
                dataPS = dataPS[0]

        # Primary Source versions
        # PCA
        v_size = dict.fromkeys(["PCA_dP", "PCA_dE", "AE"])

        if mod_pca:
            start_time = time.time()
            PS_dic["PCA"]["S"] = av.versionPS(dataPS, drift_dist="dP", nanvalue=naval, outdict=True, loadfile=False,
                                              mse_model_type=mse_model_type, tr_val_split=tr_val_split, figs_path=resultspath, 
                                              resultsfile=resultsfile, cat_threshold=cat_threshold, silent = silent, silentfig = silentfig)
            end_time = time.time()
            v_time["dP"] = end_time - start_time
            v_size["PCA_dP"] = sys.getsizeof(PS_dic["PCA"]["S"])
            print("dP summary: " + str(v_time["dP"]) + " s.; " + str(v_size["PCA_dP"]) + " b.")

            start_time = time.time()
            PS_dic["PCA"]["E"] = av.versionPS(dataPS, nanvalue=naval, drift_dist="dE", outdict=True, loadfile=False,
                                              dstitle=resultspath.split("/")[-1], mse_model_type=mse_model_type,
                                              tr_val_split=tr_val_split, figs_path=resultspath, resultsfile=resultsfile,
                                              cat_threshold = cat_threshold, silent = silent, silentfig = silentfig)
            end_time = time.time()
            v_time["dE_PCA"] = end_time - start_time
            v_size["PCA_dE"] = sys.getsizeof(PS_dic["PCA"]["E"])
            print("dE-PCA summary: " + str(v_time["dE_PCA"]) + " s.; " + str(v_size["PCA_dE"]) + " b.")

        # AE
        if mod_ae:
            start_time = time.time()
            PS_dic["AE"] = avae.versionPS(dataPS, resultspath + "aelogs/tb_logs", resultspath, nanvalue=naval,
                                          outdict=True, loadfile=False, dstitle=resultspath.split("/")[-1], mse_model_type=mse_model_type,
                                          tr_val_split=tr_val_split, noisefactor=noisefactor, figs_path=resultspath, cat_threshold = cat_threshold,
                                          resultsfile=resultsfile, silent = silent, silentfig = silentfig)
            end_time = time.time()
            v_time["dE_AE"] = end_time - start_time
            v_size["AE"] = sys.getsizeof(PS_dic["AE"])
            print("dE-AE summary: " + str(v_time["dE_AE"]) + " s.; " + str(v_size["AE"]) + " b.")

        return PS_dic, v_time, v_size

    except Exception as e:
        print(e)


def init_dicts(mod_pca, mod_ae):
    """
    Initialize dictionaries to store versioning results.

    Args:
        mod_pca (bool): Whether PCA versioning is enabled.
        mod_ae (bool): Whether AE versioning is enabled.

    Returns:
        tuple: Tuple containing the initialized dictionaries.
    """
    PS_dic = {"PCA": {"S": {}, "E": {}}, "AE": {}}
    v_size = {"dP": {}, "dE_PCA": {}, "dE_AE": {}}
    v_time = {"dP": {}, "dE_PCA": {}, "dE_AE": {}}
    v_series = {"dP": {}, "dE_PCA": {}, "dE_AE": {}}
    drift_series = {"dP": {}, "dE_PCA": {}, "dE_AE": {}}

    if not mod_pca:
        del PS_dic["PCA"], v_size["dP"], v_time["dP"], v_series["dP"], drift_series["dP"], v_size["dE_PCA"], v_time["dE_PCA"], v_series["dE_PCA"], drift_series["dE_PCA"]

    if not mod_ae:
        del PS_dic["AE"], v_size["dE_AE"], v_time["dE_AE"], v_series["dE_AE"], drift_series["dE_AE"]

    return PS_dic, drift_series, v_series, v_time, v_size


def get_PS_data(dataset, resultspath="", resultsfile="", mod_pca=True, mod_ae=True, naval=np.nan, dstitle="", pctge_PS=0.1, yname=None,
                PSfilename="psmodels.pkl", dopickle=False, tr_val_split=0.75, mse_model_type="splines", noisefactor=0.05, cat_threshold = 10,
                silent = True, silentfig = True):
    """
    Performs Primary Source (PS) data processing and versioning.

    Args:
        dataset (pd.DataFrame): Input dataset.
        resultspath (str, optional): Path to the results directory. Defaults to "".
        resultsfile (str, optional): Name of the results file. Defaults to "".
        mod_pca (bool, optional): Whether to perform PCA versioning. Defaults to True.
        mod_ae (bool, optional): Whether to perform AE versioning. Defaults to True.
        naval (float, optional): NaN value to use for imputation. Defaults to np.nan.
        dstitle (str, optional): Title for the dataset. Defaults to "".
        pctge_PS (float, optional): Percentage of Primary Source data. Defaults to 0.1.
        yname (str, optional): Name of the target variable. Defaults to None.
        PSfilename (str, optional): Name of the PS model file. Defaults to "psmodels.pkl".
        dopickle (bool, optional): Whether to pickle the results. Defaults to False.
        tr_val_split (float, optional): Train/Validation split ratio. Defaults to 0.75.
        mse_model_type (str, optional): MSE model type. Defaults to "splines".
        noisefactor (float, optional): Noise factor for AE versioning. Defaults to 0.05.

    Returns:
        tuple: Tuple containing the PS dictionaries and indices (if dopickle=True).
    """
    #print("Original data set features =======\n")
    # FRO: Added more details to the message
    print("Step [1/6]: Printing the Original Data Set Features =======")
    # FRO: Added a message that it started to generate the data set features
    print("Step [1/6]: Starting ...")
    print(dataset.describe())
    # FRO: Added a message that it finished to generate the data set features
    print("Step [1/6]: Finished.")

    # Initialize dictionaries for results
    PS_dic, _, _, v_time, _ = init_dicts(mod_pca, mod_ae)

    # Obtain clean version of the PS data set
    ind_PS_0 = dataset.index.values[0:round(pctge_PS * dataset.shape[0])]
    Xpscl_0 = av_gen.clean_ds(dataset.loc[ind_PS_0, ], mode="impute", nanvalue=naval, onehot = False, logcounts = False, threshold = cat_threshold)["dfclean"]
    ind_PS = Xpscl_0.index.values
    Xcl_PS = Xpscl_0.loc[ind_PS, ].copy()

    try:
        #print("Primary Source (initial version) data set features =======\n")
        # FRO: Added more details to the message
        print("Step [2/6]: Printing the Primary Source (PS) (initial version) data set features =======")
        # FRO: Added a message that it started to generate the data set features
        print("Step [2/6]: Starting ...")
        print(Xcl_PS.describe())
        # FRO: Added a message that it finished to generate the data set features
        print("Step [2/6]: Finished.")
    except Exception as e:
        print(e)

    # FRO: Added a message to indicate the Primary Source Features calculations has started
    print("Step [3/6]: Calculating the Primary Source Features =======")
    print("Step [3/6]: Starting ...")
    # Perform PS versioning and record metrics
    PS_dic, v_time, v_size = exp_PSinfo(Xcl_PS, PS_dic, v_time, resultspath=resultspath, mod_pca=mod_pca, mod_ae=mod_ae, naval=naval,
                                       dstitle=dstitle, mse_model_type=mse_model_type, tr_val_split=tr_val_split, noisefactor=noisefactor, 
                                       yname=yname, resultsfile=resultsfile, cat_threshold=cat_threshold, silent=silent, silentfig=silentfig)
    # print("PS: done")
    # FRO: Added a message to indicate the Primary Source Features calculations has finished
    print("Step [3/6]: Finished.")

    ps_dict = {"PS_dic": PS_dic, "v_time": v_time}

    r2_ps = {}
    if mod_pca:
        r2_ps["PCA"] = np.round(PS_dic["PCA"]["S"]["r2score"], 4)
    if mod_ae:
        r2_ps["AE"] = np.round(PS_dic["AE"]["R2_ps_opt"], 4)

    r2_ps = pd.DataFrame(r2_ps, index=[dstitle])
    v_time = pd.DataFrame(v_time, index=[dstitle])
    v_size = pd.DataFrame(v_size, index=[dstitle])

    with pd.ExcelWriter(resultspath + resultsfile + '-PS-info.xlsx') as writer:
        r2_ps.to_excel(writer, sheet_name="r2")
        v_time.to_excel(writer, sheet_name="time")
        v_size.to_excel(writer, sheet_name="size")

    if dopickle:
        file1 = open(resultspath + "-dict-" + PSfilename, 'wb')
        pickle.dump(ps_dict, file1)
        file1.close()

        file2 = open(resultspath + "-ind-" + PSfilename, 'wb')
        pickle.dump(ind_PS, file2)
        file2.close()

        pckl_files = [file1, file2]
        return ps_dict, ind_PS, pckl_files
    else:
        return ps_dict, ind_PS

def filter_commonvars(X1, X2):
    """
    Filters out common variables between two datasets.

    Args:
        X1 (pd.DataFrame): First dataset.
        X2 (pd.DataFrame): Second dataset.

    Returns:
        tuple: Tuple containing the filtered datasets and removed variables.
    """
    empty_rows1 = X1.isna().sum(axis=1) == np.shape(X1)[1]
    empty_rows2 = X2.isna().sum(axis=1) == np.shape(X2)[1]

    empty_cols_1 = X1.isna().sum(axis=0) == np.shape(X1)[0]
    empty_cols_2 = X2.isna().sum(axis=0) == np.shape(X2)[0]

    rem_vars = list(X1.columns.values[empty_cols_1]) + list(set(X2.columns.values[empty_cols_2]) - set(X1.columns.values[empty_cols_1]))

    X1_new = X1.drop(columns=rem_vars, index=X1.index.values[empty_rows1])
    X2_new = X2.drop(columns=rem_vars, index=X2.index.values[empty_rows2])

    return X1_new, X2_new, rem_vars


def vmetrics_rep(X, pcamodPS=None, aemodPS=None, Y=None, nrep=50, naval=np.nan, dstitle="", resultspath="", cat_threshold=10, silent=True, silentfig=True):
    """
    Computes versioning metrics for a given dataset using repetition-based evaluation.

    Args:
        X (pd.DataFrame): Input dataset.
        pcamodPS (dict, optional): Dictionary containing PCA models for PS versioning. Defaults to None.
        aemodPS (dict, optional): Dictionary containing AE models for PS versioning. Defaults to None.
        Y (pd.DataFrame, optional): Target variable. Defaults to None.
        nrep (int, optional): Number of repetitions. Defaults to 50.
        naval (float, optional): NaN value to use for imputation. Defaults to np.nan.
        dstitle (str, optional): Title for the dataset. Defaults to "".
        resultspath (str, optional): Path to the results directory. Defaults to "".

    Returns:
        dict: Dictionary containing the versioning metrics.
    """
    rep_metrics = {}

    if pcamodPS is not None:
        if np.round(0.2*np.shape(X)[0]) < 1:
            rep_metrics["time_dP"] = np.nan
            rep_metrics["dP"] = np.nan
            rep_metrics["vP"] = np.nan
        else:
            try:
                start_time = time.time()
                __, vD_pca_s_dict = av.versionDER(X, pcamodPS["S"], outdict=True, nanvalue=naval, loadfile=False, dstitle=dstitle, cat_threshold=cat_threshold,
                                                silent=silent, silentfig=silentfig)
                end_time = time.time()
                rep_metrics["time_dP"] = end_time - start_time
                rep_metrics["dP"] = int(vD_pca_s_dict["data_drift"])
                rep_metrics["vP"] = vD_pca_s_dict["vtag"]
            except Exception as e:
                rep_metrics["time_dP"] = np.nan
                rep_metrics["dP"] = np.nan
                rep_metrics["vP"] = np.nan
                print(e)

        try:
            start_time = time.time()
            __, vD_pca_e_dict = av.versionDER(X, pcamodPS["E"], drift_dist="dE", outdict=True, loadfile=False, nanvalue=naval, dstitle=dstitle, 
                                              cat_threshold = cat_threshold, silent=silent, silentfig=silentfig)
            end_time = time.time()
            rep_metrics["time_dEPCA"] = end_time - start_time
            rep_metrics["dEPCA"] = int(vD_pca_e_dict["data_drift"])
            rep_metrics["vEPCA"] = vD_pca_e_dict["vtag"]
        except Exception as e:
            rep_metrics["time_dEPCA"] = np.nan
            rep_metrics["dEPCA"] = np.nan
            rep_metrics["vEPCA"] = np.nan
            print(e)

    if aemodPS is not None:
        try:
            start_time = time.time()
            vD_e_e_dict = avae.versionDER(X, aemodPS, resultspath + "aelogs/tb_logs", resultspath, outdict=True, loadfile=False,
                                          nanvalue=naval, dstitle=dstitle, cat_threshold = cat_threshold, silent=silent, silentfig=silentfig)
            end_time = time.time()
            rep_metrics["time_dEAE"] = end_time - start_time
            rep_metrics["dEAE"] = int(vD_e_e_dict["data_drift"])
            rep_metrics["vEAE"] = vD_e_e_dict["vtag"]
        except Exception as e:
            rep_metrics["time_dEAE"] = np.nan
            rep_metrics["dEAE"] = np.nan
            rep_metrics["vEAE"] = np.nan
            print(e)

    return rep_metrics


def vmetrics_cv(X, pcamodPS=None, aemodPS=None, kf=KFold(n_splits=10), naval=np.nan, dstitle="", resultspath="", cat_threshold = 10,
                silent=True, silentfig=True):
    """
    Computes versioning metrics for a given dataset using cross-validation.

    Args:
        X (pd.DataFrame): Input dataset.
        pcamodPS (dict, optional): Dictionary containing PCA models for PS versioning. Defaults to None.
        aemodPS (dict, optional): Dictionary containing AE models for PS versioning. Defaults to None.
        kf (KFold, optional): Cross-validation object. Defaults to KFold(n_splits=10).
        naval (float, optional): NaN value to use for imputation. Defaults to np.nan.
        dstitle (str, optional): Title for the dataset. Defaults to "".
        resultspath (str, optional): Path to the results directory. Defaults to "".

    Returns:
        dict: Dictionary containing the versioning metrics.
    """
    if pcamodPS is not None:
        cv_dP, time_dP, cv_dEPCA, time_dEPCA = [], [], [], []
    if aemodPS is not None:
        cv_dEAE, time_dEAE = [], []

    for train, test in kf.split(X):
        Xtr = X.iloc[train, :]

        if pcamodPS is not None:
            if np.round(0.2*np.shape(X)[0]) < 1:
                time_dP.append(np.nan)
                cv_dP.append(np.nan)
            else:
                try:
                    start_time = time.time()
                    vD_pca_s_dict = av.versionDER(Xtr, pcamodPS["S"], outdict=True, nanvalue=naval, loadfile=False, dstitle=dstitle, cat_threshold = cat_threshold,
                                                silent=silent, silentfig=silentfig)
                    end_time = time.time()
                    time_dP.append(end_time - start_time)
                    cv_dP.append(vD_pca_s_dict["dpatch"])
                except Exception as e:
                    time_dP.append(np.nan)
                    cv_dP.append(np.nan)
                    print(e)

            try:
                start_time = time.time()
                vD_pca_e_dict = av.versionDER(Xtr, pcamodPS["E"], drift_dist="dE", outdict=True, loadfile=False, nanvalue=naval, dstitle=dstitle, 
                                              cat_threshold = cat_threshold, silent=silent, silentfig=silentfig)
                end_time = time.time()
                time_dEPCA.append(end_time - start_time)
                cv_dEPCA.append(np.round(vD_pca_e_dict["dpatch"], 2))
            except Exception as e:
                time_dEPCA.append(np.nan)
                cv_dEPCA.append(np.nan)
                print(e)

        if aemodPS is not None:
            try:
                start_time = time.time()
                [vD_ae_e, vD_e_e_dict] = avae.versionDER(Xtr, aemodPS, resultspath + "aelogs/tb_logs", resultspath, drift_dist = "dE", outdict=True, loadfile=False,
                                                         nanvalue=naval, dstitle=dstitle, cat_threshold = cat_threshold, silent=silent, silentfig=silentfig)
                end_time = time.time()
                time_dEAE.append(end_time - start_time)
                cv_dEAE.append(vD_e_e_dict["dpatch"])
            except Exception as e:
                time_dEAE.append(np.nan)
                cv_dEAE.append(np.nan)
                print(e)

    cv_metrics = {}

    if aemodPS is not None:
        cv_metrics["time_dEAE"] = time_dEAE
        cv_metrics["dEAE"] = cv_dEAE
    if pcamodPS is not None:
        cv_metrics["time_dEPCA"] = time_dEPCA
        cv_metrics["dEPCA"] = cv_dEPCA
        cv_metrics["time_dP"] = time_dP
        cv_metrics["dP"] = cv_dP

    return cv_metrics

def do_exp(dataset, ps_dict, resultspath="", mod_pca=True, mod_ae=True, naval=np.nan, batchsize_pctges=[1],
           n_reps=100, mode_der=None, rm_indices=[], dstitle="", n_imp_iter=10, tr_pctges=[1],
           kfolds=1, modetr="cbrt", resultsfile = "", demo = False, fixedvariables = [], cat_threshold=2,
           silent=True, silentfig=True):
    """
    Perform an experiment to compute versioning metrics for different derivations.

    Args:
        dataset (pd.DataFrame): Input dataset.
        ps_dict (dict): Dictionary containing the PCA and AE models for versioning.
        resultspath (str, optional): Path to the results directory. Defaults to "".
        mod_pca (bool, optional): Flag indicating whether to perform PCA-based versioning. Defaults to True.
        mod_ae (bool, optional): Flag indicating whether to perform AE-based versioning. Defaults to True.
        naval (float, optional): NaN value to use for imputation. Defaults to np.nan.
        batchsize_pctges (list, optional): List of batch size percentages. Defaults to [1].
        n_reps (int, optional): Number of repetitions. Defaults to 100.
        mode_der (str, optional): Derivation mode. Defaults to None.
        rm_indices (list, optional): List of indices to remove. Defaults to [].
        dstitle (str, optional): Title for the dataset. Defaults to "".
        n_imp_iter (int, optional): Number of imputation iterations. Defaults to 10.
        plot_lines (bool, optional): Flag indicating whether to plot lines. Defaults to False.
        tr_pctges (list, optional): List of transformation percentages. Defaults to [1].
        kfolds (int, optional): Number of folds for cross-validation. Defaults to 1.
        modetr (str, optional): Transformation mode. Defaults to "cbrt".
        resultsfile (str, optional): Name of the experimnet's results file. Defaults to "".
        demo (bool, optional): Flag indicating if the experiment is a demo or not. Defaults to False.
        fixedvariables (list, optional): List of variables from the PS model that must be kept in the dataset. Defaults to [].
        cat_threshold (int, optional): Maximum number of categories to consider a variable categorical. Defaults to 10.
        silent (bool, optional): Flag indicating if messages should be silenced on screen or not. Defaults to True.
        silentfig (bool, optional): Flag indicating if figures should be silenced on screen or not. Defaults to True.

    Returns:
        tuple: Tuple containing the performance series, d_patch series, v series, v time, and v size.
    """
    # Set seed for reproducibility
    # random.seed(42)

    # Initialize dictionaries allocating the results
    _, d_patch_series, v_series, _, v_size = init_dicts(mod_pca, mod_ae)
    PS_dic = ps_dict["PS_dic"]
    v_time = ps_dict["v_time"]

    dataset = av_gen.clean_ds(dataset, mode="impute", nanvalue=naval, delete_null_var=False, fixedvariables=fixedvariables, 
                              logcounts=False, onehot=False)["dfclean"]

    # Store the results
    performance_series = {"dP_cv": {}, "dEPCA_cv": {}, "dEAE_cv": {}, "tP_cv": {}, "tEPCA_cv": {}, "tEAE_cv": {}}

    if n_reps > 1:
        kfolds = 1
        kfoldspart = None
        itermode = "reps"
    elif kfolds > 1:
        n_reps = 1
        kfoldspart = KFold(n_splits=kfolds)
        itermode = "kfoldcv"
    else:
        itermode = "stop"

    # Start simulating derivations
    batch_d_pca, batch_d_pca_e, batch_v_pca, batch_v_pca_e, batch_t_pca, batch_t_pca_e = [], [], [], [], [], []
    batch_d_ae_e, batch_v_ae_e, batch_t_ae_e = [], [], []

    if np.size(tr_pctges) == 1:
        tr_pctges = [tr_pctges]
    
    if mode_der == "trans_cols":
        random.seed(42)

    for k_tr in tr_pctges:
        if mode_der != "add_batch":
            print("New versions, level " + str(k_tr) + "- start")
        for k_rm in batchsize_pctges:
            if mode_der == "add_batch":
                print("New versions, level " + str(k_rm) + "- start")
            if mod_pca:
                d_pca, d_pca_e, v_pca, v_pca_e, t_pca, t_pca_e = [], [], [], [], [], []
            if mod_ae:
                d_ae_e, v_ae_e, t_ae_e = [], [], []
            
            if re.search("batch", mode_der) is not None:
                n = int(k_rm * len(dataset))
                m = int(k_rm * len(dataset)) - 1
                if demo:
                    Xbatch_list = [dataset.iloc[:n, :]]
                else:
                    Xbatch_list = [dataset.iloc[i:i + n, :] for i in range(0, len(dataset) - m, n - m)]
                n_reps = len(Xbatch_list)
                if n>0 and np.round(0.2*n)<1:
                    print("There are not enough samples to obtain the dP metric with this setup.")
                elif n==0:
                    pred_metrics = {"time_dP":np.repeat(np.nan, n_reps), "dP":np.repeat(np.nan, n_reps), "vP":np.repeat(np.nan, n_reps), 
                                    "time_dEPCA":np.repeat(np.nan, n_reps), "dEPCA":np.repeat(np.nan, n_reps), "vEPCA":np.repeat(np.nan, n_reps), 
                                    "time_dEAE":np.repeat(np.nan, n_reps), "dEAE":np.repeat(np.nan, n_reps), "vEAE":np.repeat(np.nan, n_reps)}
                    
                    batch_t_pca.append(pred_metrics["time_dP"])
                    batch_d_pca.append(pred_metrics["dP"])
                    batch_v_pca.append(pred_metrics["vP"])

                    batch_t_pca_e.append(pred_metrics["time_dEPCA"])
                    batch_d_pca_e.append(pred_metrics["dEPCA"])
                    batch_v_pca_e.append(pred_metrics["vEPCA"])

                    batch_t_ae_e.append(pred_metrics["time_dEAE"])
                    batch_d_ae_e.append(pred_metrics["dEAE"])
                    batch_v_ae_e.append(pred_metrics["vEAE"])
                    print("There are not enough samples to run the experiments with this setup. Passing to the next level")
                    continue 
            else:
                ini_batch = random.randint(0, len(dataset) - int(k_rm * len(dataset)))
                ind_level = list(range(ini_batch, ini_batch + int(k_rm * len(dataset))))
                Xbatch = dataset[PS_dic["PCA"]["S"]["vbles_in"]].iloc[ind_level, :].copy()

            for k_rep in range(n_reps):
                if mode_der == "add_batch":
                    X_DER = Xbatch_list[k_rep]
                    title2 = " batch size"
                    xlabel_str = "Mem. size of added batch (%)"
                elif mode_der == "trans_cols":
                    X_DER = trans_cols_exp(Xbatch, level_artifact=k_tr, mode=modetr)
                    xlabel_str = "Mem. size of transformation (%)"
                    title2 = " pctge of transformed columns"
                if mode_der == "add_shifted_batch":
                    X_DER = trans_cols_exp(Xbatch_list[k_rep], level_artifact=1)
                    xlabel_str = "Mem. size of added batch (%)"
                    title2 = " batch size"
                if mode_der == "add_rw_shifted_batch":
                    X_DER = trans_rows_exp(Xbatch_list[k_rep])
                    xlabel_str = "Mem. size of added batch (%)"
                    title2 = " batch size"
                elif mode_der == "imp_md_uv":
                    X_DER, __ = miss_imp_exp(Xbatch, mode="uv", level_artifact=k_tr)
                    xlabel_str = "Mem. size of imputed entries (%)"
                    title2 = " pctge of imputed entries"
                elif mode_der == "imp_md_mv":
                    X_DER, __ = miss_imp_exp(Xbatch, mode="mv", level_artifact=k_tr, n_imp_iter=n_imp_iter)
                    xlabel_str = "Mem. size of imputed entries (%)"
                    title2 = " pctge of imputed entries"
                elif mode_der == "imp_md_knn":
                    X_DER, __ = miss_imp_exp(Xbatch, mode="knn", level_artifact=k_tr)
                    xlabel_str = "Mem. size of imputed entries (%)"
                    title2 = " pctge of imputed entries"
                if mode_der == "rem_rows_rnd":
                    X_DER = rem_rows_exp(Xbatch, mode="rnd", level_artifact=k_tr)
                    N_DER = len(X_DER)
                    xlabel_str = "Mem. size of deletion (%)"
                    ind_level = list(range(int(N_DER)))
                    title2 = " pctge of random downsampling"
                elif mode_der == "rem_rows_decimate":
                    X_DER = rem_rows_exp(Xbatch, mode="undersampling", rm_indices=[], level_artifact=k_tr)
                    N_DER = len(X_DER)
                    ind_level = list(range(int(N_DER)))
                    xlabel_str = "Mem. size of deletion (%)"
                    title2 = " pctge of downsampling"
                elif mode_der == "rem_rows_set":
                    X_DER = rem_rows_exp(Xbatch, mode="set", rm_indices=rm_indices, level_artifact=k_tr)
                    N_DER = len(X_DER)
                    ind_level = list(range(int(N_DER)))
                    xlabel_str = "Mem. size of deletion (%)"
                    title2 = " pctge of filtered records"
                elif mode_der == "rem_rows_out":
                    X_DER = rem_rows_exp(Xbatch, mode="out", level_artifact=k_tr)
                    N_DER = len(X_DER)
                    ind_level = list(range(int(N_DER)))
                    title2 = " pctge of filtered records"
                    xlabel_str = "Mem. size of deletion (%)"
                elif mode_der == "trans_rows":
                    X_DER = trans_rows_exp(Xbatch)
                    xlabel_str = "Mem. size of transformation (%)"
                    title2 = " pctge of transformed records"

                if demo:
                    return X_DER
                else:
                    if itermode == "kfoldcv":
                        pred_metrics = vmetrics_cv(X_DER, pcamodPS=PS_dic["PCA"], aemodPS=PS_dic["AE"], kf=kfoldspart, naval=naval, 
                                                    dstitle=dstitle, resultspath=resultspath, silent = silent, silentfig = silentfig, cat_threshold=cat_threshold)
                    elif itermode == "reps":
                        pred_metrics = vmetrics_rep(X_DER, pcamodPS=PS_dic["PCA"], aemodPS=PS_dic["AE"],naval=naval, dstitle=dstitle, 
                                                    resultspath=resultspath, silent = silent, silentfig = silentfig, cat_threshold=cat_threshold)
                    else:
                        pred_metrics = {"time_dP":np.nan, "dP":np.nan, "vP":np.nan, "time_dEPCA":np.nan, "dEPCA":np.nan, "vEPCA":np.nan, 
                                        "time_dEAE":np.nan, "dEAE":np.nan, "vEAE":np.nan}

                t_pca.append(pred_metrics["time_dP"])
                d_pca.append(pred_metrics["dP"])
                v_pca.append(pred_metrics["vP"])

                t_pca_e.append(pred_metrics["time_dEPCA"])
                d_pca_e.append(pred_metrics["dEPCA"])
                v_pca_e.append(pred_metrics["vEPCA"])

                t_ae_e.append(pred_metrics["time_dEAE"])
                d_ae_e.append(pred_metrics["dEAE"])
                v_ae_e.append(pred_metrics["vEAE"])

            if np.size(batchsize_pctges) > 1:
                batch_t_pca.append(t_pca)
                batch_d_pca.append(d_pca)
                batch_v_pca.append(v_pca)

                batch_t_pca_e.append(t_pca_e)
                batch_d_pca_e.append(d_pca_e)
                batch_v_pca_e.append(v_pca_e)

                batch_t_ae_e.append(t_ae_e)
                batch_d_ae_e.append(d_ae_e)
                batch_v_ae_e.append(v_ae_e)

            if mode_der == "add_batch":
                print("New versions, level " + str(k_rm) + "- end")

        if np.size(tr_pctges) > 1:
            batch_t_pca.append(t_pca)
            batch_d_pca.append(d_pca)
            batch_v_pca.append(v_pca)

            batch_t_pca_e.append(t_pca_e)
            batch_d_pca_e.append(d_pca_e)
            batch_v_pca_e.append(v_pca_e)

            batch_t_ae_e.append(t_ae_e)
            batch_d_ae_e.append(d_ae_e)
            batch_v_ae_e.append(v_ae_e)
            
        if mode_der != "add_batch":
            print("New versions, level " + str(k_tr) + "- end")

    if len(tr_pctges) > 1:
        col_names = tr_pctges
    elif len(batchsize_pctges) > 1:
        col_names = batchsize_pctges
    else:
        col_names = ["p100"]

    performance_series["dP"] = pd.DataFrame(batch_d_pca, columns=list(range(len(batch_d_pca[0]))), index=["level " + str(x) for x in col_names])
    performance_series["dEPCA"] = pd.DataFrame(batch_d_pca_e, columns=list(range(len(batch_d_pca_e[0]))), index=["level " + str(x) for x in col_names])
    performance_series["dEAE"] = pd.DataFrame(batch_d_ae_e, columns=list(range(len(batch_d_ae_e[0]))), index=["level " + str(x) for x in col_names])

    performance_series["tP"] = pd.DataFrame(batch_t_pca, columns=list(range(len(batch_t_pca[0]))), index=["level " + str(x) for x in col_names])
    performance_series["tEPCA"] = pd.DataFrame(batch_t_pca_e, columns=list(range(len(batch_t_pca_e[0]))), index=["level " + str(x) for x in col_names])
    performance_series["tEAE"] = pd.DataFrame(batch_t_ae_e, columns=list(range(len(batch_t_ae_e[0]))), index=["level " + str(x) for x in col_names])

    performance_series["vP"] = pd.DataFrame(batch_v_pca, columns=list(range(len(batch_v_pca[0]))), index=["level " + str(x) for x in col_names])
    performance_series["vEPCA"] = pd.DataFrame(batch_v_pca_e, columns=list(range(len(batch_v_pca_e[0]))), index=["level " + str(x) for x in col_names])
    performance_series["vEAE"] = pd.DataFrame(batch_v_ae_e, columns=list(range(len(batch_v_ae_e[0]))), index=["level " + str(x) for x in col_names])

    # FRO: Message to finish a process
    print("DER: done")
    exp_write_results(performance_series, resultspath + resultsfile, mod_pca, mod_ae)
    return performance_series, d_patch_series, v_series, v_time, v_size

def rem_rows_exp(dataset, mode = "rnd", rm_indices = [], level_artifact = 0.01):
    """
    Removes rows from the dataset based on the specified mode.

    Args:
        dataset (pd.DataFrame): The input dataset.
        mode (str, optional): The mode of row removal. Default is "rnd" (random).
        rm_indices (list, optional): A list of indices to remove when mode is "set".
        level_artifact (float, optional): The level of artifact to be introduced, determining the proportion of rows to remove.

    Returns:
        pd.DataFrame: The modified dataset with rows removed.
    """
    # Set seed for reproducibility
    random.seed(42)

    if isinstance(level_artifact, list):
        level_artifact = level_artifact[0]
    if mode == "rnd": 
        ind_rep = random.sample(list(dataset.index.values), int(level_artifact*len(dataset)))
        Xnew = dataset.copy()
        Xnew = Xnew.drop(index = ind_rep)
    if mode == "undersampling":
        ind_rep = dataset.index.values[::int(1 / level_artifact)]
        Xnew =  dataset.copy().loc[ind_rep,:]
    elif mode == "set":
        ind_rep = random.sample(rm_indices, int(level_artifact*len(rm_indices)))
        Xnew = dataset.copy()
        Xnew = Xnew.drop(index = ind_rep)
    elif mode == "out":
        Xnum = dataset.select_dtypes("number")
        Xsc = pd.DataFrame(StandardScaler().fit_transform(Xnum), index=Xnum.index.values, columns = Xnum.columns.values)
        q3, q2, q1 = np.nanpercentile(Xsc, [75 , 50, 25], axis=0)
        viqr = q3 - q1
        # Non-parammetric (not assuming normality, just the intrinsic features of the variables)
        ind_out = list(dataset.index.values[np.sum(Xsc.apply(lambda x: (x < (q1 - 1.5*viqr)) | (x > (q3 + 1.5*viqr)), axis=1), axis = 1) > max([level_artifact*np.shape(Xsc)[1], 1])])
        Xnew = dataset.copy()
        Xnew = Xnew.drop(index = ind_out)
    return(Xnew)

def trans_cols_exp(dataset, level_artifact = None, mode = "cbrt"):
    """
    Transforms columns of the dataset based on the specified mode.

    Args:
        dataset (pd.DataFrame): The input dataset.
        level_artifact (float, optional): The level of artifact to be introduced, determining the proportion of columns to transform.
        mode (str, optional): The transformation mode. Default is "cbrt" (cubic root).

    Returns:
        pd.DataFrame: The modified dataset with transformed columns.
    """
    # Set seed for reproducibility
    # random.seed(42)

    if isinstance(level_artifact, list):
        level_artifact = level_artifact[0]
    ind_rep = np.sort(random.sample(range(len(list(dataset.columns.values))), int(max([level_artifact*np.shape(dataset)[1], 1]))))
    Xnew = dataset.copy()
    Xnew.iloc[:,ind_rep] = nonlintrans(dataset.iloc[:,ind_rep], mode = mode)
    return(Xnew)

def trans_rows_exp(dataset):
    """
    Transforms rows of the dataset by normalizing each row.

    Args:
        dataset (pd.DataFrame): The input dataset.

    Returns:
        pd.DataFrame: The modified dataset with transformed rows.
    """
    Xnew = dataset.copy().apply(lambda x: x / np.sum(x), axis = 1)
    return(Xnew)

def miss_imp_exp(dataset, mode = "mv", level_artifact = None, imp_strat = "mean", nmax_imp_iter = 10, nmax_imp_neigh = 10):
    """
    Performs missing data imputation on the dataset based on the specified mode.

    Args:
        dataset (pd.DataFrame): The input dataset.
        mode (str, optional): The imputation mode. Default is "mv" (multivariate).
        level_artifact (float, optional): The level of artifact to be introduced, determining the proportion of missing entries to impute.
        imp_strat (str, optional): The imputation strategy used when mode is "uv" (univariate). Default is "mean".
        nmax_imp_iter (int, optional): The maximum number of imputation iterations when mode is "mv" (multivariate). Default is 10.
        nmax_imp_neigh (int, optional): The maximum number of nearest neighbors to consider when mode is "knn" (k-nearest neighbors). Default is 10.

    Returns:
        pd.DataFrame: The modified dataset with missing data imputed.
        str: The imputation mode used.
    """
    # Set seed for reproducibility
    random.seed(42)

    if isinstance(level_artifact, list):
        level_artifact = level_artifact[0]
    ind_rep = random.sample(range(0,np.size(dataset)), int(level_artifact*np.size(dataset)))
    xvec = np.reshape(np.ravel(dataset.copy()), np.size(dataset))
    if xvec.dtype == np.int_: xvec = xvec.astype(np.float_)
    xvec[ind_rep] = np.nan
    Xnew = pd.DataFrame(np.reshape(xvec, np.shape(dataset)).copy(), index=dataset.index.values, columns=dataset.columns.values)
    # Impute with mean or with 
    if mode == "uv":
        imp = SimpleImputer(missing_values=np.nan, strategy= imp_strat)
        imp.set_params(keep_empty_features=True)
        Ximp = pd.DataFrame(imp.fit_transform(Xnew), index=Xnew.index.values, columns=Xnew.columns.values)
        imp_mode = mode + '-' + imp_strat
    elif mode == "mv":
        
        if np.shape(Xnew)[1] > 100:
            imp = IterativeImputer(max_iter=nmax_imp_iter, random_state=0, verbose=2, estimator=RandomForestRegressor(n_estimators=4,
                                                                                                                        max_depth=10,
                                                                                                                        bootstrap=True,
                                                                                                                        max_samples=0.5,
                                                                                                                        n_jobs=2,
                                                                                                                        random_state=0))
        else:
            if np.shape(Xnew)[1] >= np.shape(Xnew)[0]:
                imp = IterativeImputer(max_iter=nmax_imp_iter, random_state=0, verbose=2, n_nearest_features=np.round(0.8*np.shape(Xnew)[0]))
            else:
                imp = IterativeImputer(max_iter=nmax_imp_iter, random_state=0, verbose=2)

        Ximp = pd.DataFrame(imp.fit_transform(Xnew), index=Xnew.index.values, columns=Xnew.columns.values)
        imp_mode = mode
    elif mode == "knn":
        best_k = optimize_knn_imputation(Xnew, range(1, nmax_imp_neigh, 2))
        imp = KNNImputer(n_neighbors=best_k, weights="uniform")
        Ximp = pd.DataFrame(imp.fit_transform(Xnew), index=Xnew.index.values, columns=Xnew.columns.values)
        imp_mode = mode
    return(Ximp, imp_mode)   
         
def optimize_knn_imputation(X, k_values):
    """
    Optimizes the number of nearest neighbors for KNN imputation.

    Args:
        X (pd.DataFrame): The input dataset.
        k_values (list): List of candidate values for the number of nearest neighbors.

    Returns:
        int: The optimized number of nearest neighbors.
    """
    # Set seed for reproducibility
    random.seed(42)
    
    p_missing = X.isna().sum().sum()/np.size(X)
    
    ind_rep = random.sample(range(0,np.size(X)), int(p_missing*np.size(X)))
    
    rmse = []
    for k in k_values:
        imp = KNNImputer(n_neighbors=k, weights="uniform")
        Ximp = pd.DataFrame(imp.fit_transform(X), index=X.index.values, columns=X.columns.values)

        xvec = np.reshape(np.ravel(Ximp.copy()), np.size(X))
        if xvec.dtype == np.int_: xvec = xvec.astype(np.float_)
        xvec[ind_rep] = np.nan
        Xna_k = pd.DataFrame(np.reshape(xvec, np.shape(Ximp)).copy(), index=X.index.values, columns=X.columns.values)
    
        
        Xtest_k = pd.DataFrame(imp.transform(Xna_k), index=X.index.values, columns=X.columns.values)
        
        rmse.append(np.sqrt(np.nanmean((np.square(Xtest_k - Xna_k)).to_numpy(), axis=None)))
    k_list = list(k_values)
    best_k = k_list[np.where(rmse==min(rmse))[0][0]]
    return best_k

def nonlintrans(X, mode = "cbrt", epsilon = 10e-4):
    """
    Performs a nonlinear transformation on the input dataset.

    Args:
        X (pd.DataFrame): The input dataset.
        mode (str, optional): The transformation mode. Default is "cbrt" (cubic root).
        epsilon (float, optional): The epsilon value used for logarithmic transformation. Default is 10e-4.

    Returns:
        pd.DataFrame: The transformed dataset.
    """
    Xnew = X.copy()
    xneg = Xnew.apply(lambda x: any(x <= 0), axis = 1)
    if mode == "cbrt":
        Xnew = np.cbrt(Xnew)
    elif mode == "log":
        Xnew.loc[~xneg] = np.log(Xnew.loc[~xneg])
        Xnew.loc[xneg] = np.log(Xnew.loc[~xneg] + np.abs(np.min(Xnew.loc[xneg])) + epsilon)
    return(Xnew)

def extract_seasonality_trend(data, freq):
    """
    Extracts seasonal and trend components from a multivariate time series dataset.

    Args:
        data (pd.DataFrame): The multivariate time series dataset.
        freq (int): The frequency of the data (e.g., 7 for weekly data, 12 for monthly data).

    Returns:
        seasonal (pd.DataFrame): The seasonal components of the dataset.
        trend (pd.DataFrame): The trend components of the dataset.
    """
    
    if np.sum(data.isnull().any(axis=0))==np.shape(data)[1]:
        data = data.interpolate(method='linear')
    seasonal = pd.DataFrame(columns = data.columns)
    trend = pd.DataFrame(columns = data.columns)
    for column in data.columns:
        if data[column].isnull().any():
            seasonal[column] = 0
            trend[column] = 0
        else:
            decomposition = seasonal_decompose(data[column], period=freq)
            seasonal[column] = decomposition.seasonal
            trend[column] = decomposition.trend

    return seasonal, trend

def convert_tuples_to_dates(tuple_list):
    """
    Converts a list of tuples into a list of dates.

    Args:
        tuple_list (list): The list of tuples.

    Returns:
        dates_list (list): The list of dates.
    """
    dates_list = []
    for tpl in tuple_list:
        if isinstance(tpl, tuple):
            datex = tpl[0].date()
            if not isinstance(tpl[1], str):
                time_str = tpl[1].strftime("%H:%M")
            else:
                time_str = tpl[1]
            datex = tpl[0].date()
            hours, minutes = map(int, time_str.split(':'))
            time_delta = timedelta(hours=hours, minutes=minutes)
            complete_timestamp = datetime.combine(datex, (datetime.min + time_delta).time())
            dates_list.append(complete_timestamp)
        else:
            dates_list.append(tpl)
    
    return dates_list

def plot_time_components(dataf, freq, xplot = None, path_figs = "figures/", dsname = "", silent = True, silentfig = True):
    """
    Plots the original data, trend component, and seasonal component of a time series.

    Args:
        dataf (pd.DataFrame): The input time series data.
        freq (str): The frequency of the time series (e.g., 'D' for daily, 'M' for monthly).
        xplot (array-like, optional): The x-axis values for plotting. Default is None.
        path_figs (str, optional): The path to save the generated figures. Default is "../../Reports/series/".
        dsname (str, optional): The name of the dataset. Default is an empty string.

    Returns:
        str: A string indicating the completion of the function.
    """
    seasonal, trend = extract_seasonality_trend(dataf, freq)
    if xplot is None:
        xplot = convert_tuples_to_dates(dataf.index.values)
    # Visualize the extracted components
    for column in dataf.columns[0:]:
        if sum(trend[column]>0):
            plt.figure(figsize=(10, 6))
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, dpi=400)
            ax1.plot(xplot, dataf[column])
            ax1.set_title(f'{column} - Original Data')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Value')

            ax2.plot(xplot, trend[column])
            ax2.set_title(f'{column} - Trend Component')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Value')

            ax3.plot(xplot, seasonal[column])
            ax3.set_title(f'{column} - Seasonal Component')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Value')

            plt.tight_layout()
            if isinstance(column,tuple): column = ' '.join(column)
            fig.savefig(path_figs + dsname + "-components-" + column.replace("/","-") + ".png", bbox_inches="tight")
            if not silentfig:
                plt.show()
                # FRO Added the following line to remove the need to press a key
                # plt.pause(0.1)
            plt.close()

def plot_time_components_combined(dataf_ps, dataf_rev, freq, xplot_ps = None, xplot_rev = None, path_figs = "figures/", dsname = "", silent = True,
                                  silentfig = True):
    """
    Plots the original data, trend component, and seasonal component of two combined time series.

    Args:
        dataf_ps (pd.DataFrame): The primary time series data.
        dataf_rev (pd.DataFrame): The revised time series data.
        freq (str): The frequency of the time series (e.g., 'D' for daily, 'M' for monthly).
        xplot_ps (array-like, optional): The x-axis values for the primary time series plotting. Default is None.
        xplot_rev (array-like, optional): The x-axis values for the revised time series plotting. Default is None.
        path_figs (str, optional): The path to save the generated figures. Default is "../../Reports/series/".
        dsname (str, optional): The name of the dataset. Default is an empty string.

    Returns:
        str: A string indicating the completion of the function.
    """
    seasonal_ps, trend_ps = extract_seasonality_trend(dataf_ps, freq)
    seasonal_rev, trend_rev = extract_seasonality_trend(dataf_rev, freq)
    
    if xplot_ps is None:
        xplot_ps = convert_tuples_to_dates(dataf_ps.index.values)
    if xplot_rev is None:
        xplot_rev = convert_tuples_to_dates(dataf_rev.index.values)

    # Visualize the extracted components
    for column in dataf_ps.columns[0:]:
        if sum(trend_rev[column]>0):
            plt.figure(figsize=(10, 6))
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, dpi=400)
            ax1.plot(range(len(xplot_ps)), dataf_ps[column],color="k")
            ax1.plot(range(len(xplot_ps), len(xplot_ps) + len(xplot_rev)), dataf_rev[column],color="r",linestyle="--")
            ax1.set_title(f'{column} - Original Data')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Value')

            ax2.plot(range(len(xplot_ps)), trend_ps[column],color="k")
            ax2.plot(range(len(xplot_ps), len(xplot_ps) + len(xplot_rev)), trend_rev[column],color="r",linestyle="--")
            ax2.set_title(f'{column} - Trend Component')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Value')

            ax3.plot(range(len(xplot_ps)), seasonal_ps[column],color="k")
            ax3.plot(range(len(xplot_ps), len(xplot_ps) + len(xplot_rev)), seasonal_rev[column],color="r",linestyle="--")
            ax3.set_title(f'{column} - Seasonal Component')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Value')

            plt.tight_layout()
            if isinstance(column,tuple): column = ' '.join(column)
            fig.savefig(path_figs + dsname + "-components-" + column.replace("/","-") + ".png", bbox_inches="tight")
            if not silentfig:
                plt.show()
                # FRO: Added the following line to remove the need to press a key
                # plt.pause(0.1)
            plt.close()

def plot_time_components_div(dataf, ind_ps, ind_rev, freq, xplot_ps = None, xplot_rev = None, path_figs = "figures/", dsname = "", 
                             xtxtsize = 8, onlydate = True, silent = True, silentfig = True):
    """
    Plots the original data, trend component, and seasonal component of a divided time series.

    Args:
        dataf (pd.DataFrame): The input time series data.
        ind_ps (pd.Index): The index values corresponding to the primary time series.
        ind_rev (pd.Index): The index values corresponding to the revised time series.
        freq (str): The frequency of the time series (e.g., 'D' for daily, 'M' for monthly).
        xplot_ps (array-like, optional): The x-axis values for the primary time series plotting. Default is None.
        xplot_rev (array-like, optional): The x-axis values for the revised time series plotting. Default is None.
        path_figs (str, optional): The path to save the generated figures. Default is "../../Reports/series/".
        dsname (str, optional): The name of the dataset. Default is an empty string.
        xtxtsize (int, optional): The font size of x-axis labels. Default is 8.
        onlydate (bool, optional): Whether to display only the date in x-axis labels. Default is True.

    Returns:
        str: A string indicating the completion of the function.
    """
    seasonal, trend = extract_seasonality_trend(dataf, freq)
    
    if xplot_ps is None:
        xplot_ps = convert_tuples_to_dates(ind_ps.values)
    if xplot_rev is None:
        xplot_rev = convert_tuples_to_dates(ind_rev.values)

    if onlydate:
        if isinstance(xplot_ps[0], pd.Timestamp):
            xplot_ps = [pd.to_datetime(x).date() for x in xplot_ps]
        elif isinstance(xplot_ps[0], datetime):
            xplot_ps = [x.strftime("%m/%d/%Y" ) for x in xplot_ps]
        if isinstance(xplot_rev[0], pd.Timestamp):
            xplot_rev = [pd.to_datetime(x).date() for x in xplot_rev]
        elif isinstance(xplot_rev[0], datetime):
            xplot_rev = [x.strftime("%m/%d/%Y" ) for x in xplot_rev]

    if not isinstance(xplot_ps, list):
        xlabs = xplot_ps.to_list() + xplot_rev.to_list()
    else:
        xlabs = xplot_ps + xplot_rev
    if any(dataf.index.duplicated()):
        aa = dataf.index.duplicated()
        dataf = dataf.iloc[~aa,:]
        trend = trend.iloc[~aa,:]
        seasonal = seasonal.iloc[~aa,:]
    
    # Visualize the extracted components
    for column in dataf.columns[0:]:
        if sum(trend[column]>0):
            # plt.figure(figsize=(10, 6))
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, dpi=400, figsize=(10, 6))
            ax1.plot(range(len(xplot_ps)), dataf[column][ind_ps],color="k")
            ax1.plot(range(len(xplot_ps), len(xplot_ps) + len(xplot_rev)), dataf[column][ind_rev],color="r",linestyle="--")
            current_xticks = [int(pos)  for pos in ax1.get_xticks() if (pos >= 0 and pos < len(xlabs))]
            current_labs = [xlabs[pos] for pos in current_xticks]
            ax1.set_xticks(current_xticks, current_labs)
            ax1.xaxis.set_tick_params(labelsize=xtxtsize)
            ax1.set_title(f'{column} - Original Data')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Value')

            ax2.plot(range(len(xplot_ps)), trend[column][ind_ps],color="k")
            ax2.plot(range(len(xplot_ps), len(xplot_ps) + len(xplot_rev)), trend[column][ind_rev],color="r",linestyle="--")
            current_xticks = [int(pos)  for pos in ax2.get_xticks() if (pos >= 0 and pos < len(xlabs))]
            current_labs = [xlabs[pos] for pos in current_xticks]
            ax2.set_xticks(current_xticks, current_labs)
            ax2.set_title(f'{column} - Trend Component')
            ax2.xaxis.set_tick_params(labelsize=xtxtsize)
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Value')

            ax3.plot(range(len(xplot_ps)), seasonal[column][ind_ps],color="k")
            ax3.plot(range(len(xplot_ps), len(xplot_ps) + len(xplot_rev)), seasonal[column][ind_rev],color="r",linestyle="--")
            current_xticks = [int(pos)  for pos in ax3.get_xticks() if (pos >= 0 and pos < len(xlabs))]
            current_labs = [xlabs[pos] for pos in current_xticks]
            ax3.set_xticks(current_xticks, current_labs)
            ax3.set_title(f'{column} - Seasonal Component')
            ax3.xaxis.set_tick_params(labelsize=xtxtsize)
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Value')

            fig.tight_layout()
            if isinstance(column,tuple): column = ' '.join(column)
            fig.savefig(path_figs + dsname + "-components-" + column.replace("/","-") + ".png", bbox_inches="tight", dpi = 400)
            if not silentfig:
                plt.show(fig)
                # FRO Added the following line to remove the need to press a key
                #plt.pause(0.1)
            #plt.close()
            plt.close(fig)
            
    return("done")
