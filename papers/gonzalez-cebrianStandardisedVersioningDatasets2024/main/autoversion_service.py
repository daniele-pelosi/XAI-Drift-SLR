"""
Script Name: autoversion_service.py
Description: this module is a Python script designed to perform PCA-based versioning with 
data drift detection using two different approaches: PS (Primary Source) and DER (Derived). 
It defines several functions and includes options for loading datasets, cleaning data, 
performing Principal Component Analysis (PCA), and computing versioning parameters.

Author:
- Name: Alba González-Cebrián, Fanny Rivera-Ortiz, Jorge Mario Cortés-Mendoza, Adriana E. Chis, Michael Bradford, Horacio González-Vélez
- Email: Alba.Gonzalez-Cebrian@ncirl.ie

License: MIT License
- License URL: https://opensource.org/license/mit/
"""

import json
import copy
import datetime
import numpy as np
import pandas as pd
import autoversion_gen as av_gen
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis

def versionPS(df, modeclean="r&c", nanvalue=np.nan, drift_dist = "dP", nrep=50, alpha=0.05, outdict=False, loadfile=True, hname=None, usecols=None, skiprows=0, 
              dtype=float, dopickle=False, delimiter=None, respath="", decimal=".", dojson=False, dstitle = "", mse_model_type = "reg", tr_val_split = 0.8,
              steps_list = [5, 10, 15, 30, 50], figs_path = "", resultsfile = "", cat_threshold = 10, silent = False, silentfig = True):
    """
    Function to perform PCA-based versioning with data drift detection using the PS (Primary Source) approach.

    Args:
        df (DataFrame): Input dataset.
        modeclean (str, optional): Mode for cleaning the dataset. Defaults to "r&c".
        nanvalue (float, optional): Value to represent missing values. Defaults to np.nan.
        drift_dist (str, optional): Drift distance measure. Defaults to "dP".
        nrep (int, optional): Number of repetitions for estimating confidence intervals. Defaults to 50.
        alpha (float, optional): Significance level for confidence intervals. Defaults to 0.05.
        outdict (bool, optional): Flag to indicate whether to return the output as a dictionary. Defaults to False.
        loadfile (bool, optional): Flag to indicate whether to load the dataset from a file. Defaults to True.
        hname (str, optional): Name of the file to load the dataset. Defaults to None.
        usecols (list, optional): List of columns to use from the dataset. Defaults to None.
        skiprows (int, optional): Number of rows to skip when loading the dataset. Defaults to 0.
        dtype (type, optional): Data type to use for the dataset. Defaults to float.
        dopickle (bool, optional): Flag to indicate whether to load a pickled dataset. Defaults to False.
        delimiter (str, optional): Delimiter to use when loading the dataset. Defaults to None.
        respath (str, optional): Path to the results directory. Defaults to "".
        decimal (str, optional): Decimal separator to use when loading the dataset. Defaults to ".".
        dojson (bool, optional): Flag to indicate whether to load a JSON-formatted dataset. Defaults to False.
        dstitle (str, optional): Title for the dataset. Defaults to "".
        mse_model_type (str, optional): Model type for mean squared error estimation. Defaults to "reg".
        tr_val_split (float, optional): Train-validation split ratio. Defaults to 0.8.
        steps_list (list, optional): List with numbers of time steps for time-series data. Defaults to [5, 10, 15, 30, 50].
        figs_path (str, optional): Path to save figures. Defaults to "".
        resultsfile (str, optional): Path to save the results. Defaults to "".
        cat_threshold (int, optional): integer indicating the maximum number of categories for a variable to be considered categorical. Deafaults to 10.
        silent (bool, optional): Boolean value indicating if comments are printed in the screen or not. Defaults to True.
        silentfig (bool, optional): Boolean value indicating if figures are printed in the screen during the execution or not. Defaults to True.

    Returns:
        dict or str: Dictionary or JSON-formatted string representing the PS version information.
    """

    if loadfile:
        df = av_gen.load_df(copy.deepcopy(df), hname=hname, usecols=usecols, skiprows=skiprows, dtype=dtype, dopickle=dopickle, delimiter=delimiter, respath=respath, 
                            decimal=decimal, dojson=dojson)
    
    dfps_clean = av_gen.clean_ds(df, mode=modeclean, nanvalue=nanvalue, threshold = cat_threshold, onehot=False, logcounts=False)
    PSv = version_info_pca(dfps_clean["dfclean"], mode="ps", drift_dist = drift_dist, alpha = alpha, nrep = nrep, dstitle = dstitle, mse_model_type = mse_model_type,
                           tr_val_split = tr_val_split, steps_list = steps_list, figs_path = figs_path, resultsfile = resultsfile, blocks = dfps_clean["blocks"], 
                           silent = silent, silentfig = silentfig)
    if PSv is not None:
        PSv["blocks"] = dfps_clean["blocks"]
        PSv["ohe"] = dfps_clean["ohe"]
        PSv["model_class"] = "PCA"
        PSv["vbles_in_prepro"] = dfps_clean["vbles_in_prepro"]
        Psv2 = PSv.copy()
        if drift_dist == "dE": 
            Psv2.pop("mse_model")
    return(PSv)
    
def versionDER(df, ps_version_inf, modeclean = "r&c", nanvalue = np.nan, drift_dist = "dP", outdict = False, loadfile = True, hname = None, usecols = None, skiprows = 0, 
               dtype = float, dopickle = False, delimiter = None, respath = "", decimal = ".", dojson = False, dstitle = "", cat_threshold = 10, silent = False, silentfig = True):
    """
    Function to perform PCA-based versioning with data drift detection using the DER (Derived) approach.

    Args:
        df (DataFrame): Input dataset.
        ps_version_inf (dict or str): PS version information as a dictionary or JSON-formatted string.
        modeclean (str, optional): Mode for cleaning the dataset. Defaults to "r&c".
        nanvalue (float, optional): Value to represent missing values. Defaults to np.nan.
        drift_dist (str, optional): Drift distance measure. Defaults to "dP".
        outdict (bool, optional): Flag to indicate whether to return the output as a dictionary. Defaults to False.
        loadfile (bool, optional): Flag to indicate whether to load the dataset from a file. Defaults to True.
        hname (str, optional): Name of the file to load the dataset. Defaults to None.
        usecols (list, optional): List of columns to use from the dataset. Defaults to None.
        skiprows (int, optional): Number of rows to skip when loading the dataset. Defaults to 0.
        dtype (type, optional): Data type to use for the dataset. Defaults to float.
        dopickle (bool, optional): Flag to indicate whether to load a pickled dataset. Defaults to False.
        delimiter (str, optional): Delimiter to use when loading the dataset. Defaults to None.
        respath (str, optional): Path to the results directory. Defaults to "".
        decimal (str, optional): Decimal separator to use when loading the dataset. Defaults to ".".
        dojson (bool, optional): Flag to indicate whether to load a JSON-formatted dataset. Defaults to False.
        dstitle (str, optional): Title for the dataset. Defaults to "".
        cat_threshold (int, optional): integer indicating the maximum number of categories for a variable to be considered categorical. Deafaults to 10.
        silent (bool, optional): Boolean value indicating if comments are printed in the screen or not. Defaults to True.
        silentfig (bool, optional): Boolean value indicating if figures are printed in the screen during the execution or not. Defaults to True.

    Returns:
        dict or str: Dictionary or JSON-formatted string representing the DER version information.
    """

    if loadfile:
        df = av_gen.load_df(copy.deepcopy(df), hname=hname, usecols=usecols, skiprows=skiprows, dtype=dtype, dopickle=dopickle, delimiter=delimiter, respath=respath, 
                            decimal=decimal, dojson=dojson)
    
    if isinstance(ps_version_inf, dict):
        PSv = ps_version_inf
    else: PSv = av_gen.json2dict(ps_version_inf, model_type="pca")
    
    if drift_dist == "dP":
        dfps_clean_new = av_gen.clean_ds(df, mode = "impute", nanvalue = nanvalue, ohe=ps_version_inf["ohe"], threshold = cat_threshold, onehot=False, 
                                         logcounts=False)
        DERv = version_info_pca(dfps_clean_new["dfclean"], mode = "der", ps_version_inf = PSv, drift_dist = drift_dist, dstitle = dstitle, 
                                blocks = dfps_clean_new["blocks"], silent = silent)
    elif drift_dist == "dE":
        dfps_clean_new = av_gen.clean_ds(df, mode = "impute", nanvalue = nanvalue, delete_null_var=False, ohe=ps_version_inf["ohe"], threshold = cat_threshold, 
                                         fixedvariables = list(PSv["pcamodel"]["model"].feature_names_in_), onehot=False, logcounts=False)
        DERv = version_info_pca(dfps_clean_new["dfclean"], mode = "der", ps_version_inf = PSv, drift_dist = drift_dist, dstitle = dstitle, blocks = ps_version_inf["blocks"], silent = silent)
    
    DERv_out = dict((k, DERv[k]) for k in ('vtag', 'data_drift'))
    
    if outdict: return(json.dumps(DERv_out), DERv)
    else: return(json.dumps(DERv_out))

def version_info_pca(df, mode = "ps", ps_version_inf = None, drift_dist = "dP", nrep = 50, alpha = 0.05, dstitle = "", mse_model_type = "reg", 
                     tr_val_split = 0.8, steps_list = [5, 10, 15, 30, 50], figs_path = "", resultsfile = "", blocks = None, silent = False, silentfig = True):
    """
    Function to compute version information and detect data drift using PCA.

    Args:
        df (DataFrame): Input dataset.
        mode (str, optional): Mode of operation. Defaults to "ps".
        ps_version_inf (dict or str, optional): PS version information as a dictionary or JSON-formatted string. Defaults to None.
        drift_dist (str, optional): Drift distance measure. Defaults to "dP".
        nrep (int, optional): Number of repetitions for estimating confidence intervals. Defaults to 50.
        alpha (float, optional): Significance level for confidence intervals. Defaults to 0.05.
        dstitle (str, optional): Title for the dataset. Defaults to "".
        mse_model_type (str, optional): Model type for mean squared error estimation. Defaults to "reg".
        tr_val_split (float, optional): Train-validation split ratio. Defaults to 0.8.
        steps_list (list, optional): List with numbers of time steps for time-series data. Defaults to [5, 10, 15, 30, 50].
        figs_path (str, optional): Path to save figures. Defaults to "".
        resultsfile (str, optional): Path to save the results. Defaults to "".
        blocks (list, optional): Block indicator for each variable. Deafaults to None.
        silent (bool, optional): Boolean value indicating if comments are printed in the screen or not. Defaults to True.
        silentfig (bool, optional): Boolean value indicating if figures are printed in the screen during the execution or not. Defaults to True.

    Returns:
        dict: Dictionary representing the version information.
    """

    if mode == "ps":
        # Apply k-fold (l.o.o) to obtain C.I. on the distance parameters (for each component)
        dfnum = df.select_dtypes(include=np.number)
        ps_version_inf = vparams_pca(dfnum, tr_val_split = tr_val_split, blocks = blocks)
        
        if np.shape(dfnum)[1] == 1:
            dfnum_steps, _ = av_gen.create_dataset(dfnum, len(ps_version_inf["vbles_in"]))
            dfnum = pd.DataFrame(dfnum_steps, columns=ps_version_inf["vbles_in"])
        else: dfnum = df[ps_version_inf["vbles_in"]]

        if drift_dist == "dE":
            ps_version_inf["mse_model"] = av_gen.fit_mse_model(dfnum, ps_version_inf, vparams_pca, dstitle = dstitle, mse_model_type = mse_model_type, 
                                                               figs_path = figs_path, resultsfile = resultsfile, silent = silent, silentfig = silentfig)

        ps_version_inf["data_model"] = df.dtypes
        ps_version_inf["timestamp"] = datetime.datetime.now(tz=datetime.timezone.utc)
        ps_version_inf["vtag"] = "1.0" + ".<" + ps_version_inf["timestamp"].strftime("%m/%d/%Y - %H:%M:%S") + ">"
        ps_version_inf["n"] = len(dfnum) - 1
        return(ps_version_inf)

    if mode == "der":
        dfold = df.select_dtypes(include=np.number)
        
        if np.shape(dfold)[1] == 1:
            dfnum_steps, _ = av_gen.create_dataset(dfold, len(ps_version_inf["vbles_in"]))
            df = pd.DataFrame(dfnum_steps, columns = ps_version_inf["vbles_in"])
        else: df = dfold
                        
        # Obtain distance and check with the intervals
        
        v_major_ps, v_minor_ps, v_patch_ps = ps_version_inf["vtag"].split(".")

        if drift_dist == "dP":
            Dinfo = vparams_pca(df, steps_list = steps_list, blocks = ps_version_inf["blocks"])
            dP = s_distance(df, ps_version_inf, Rinfo=Dinfo)["data_drift"]
            
            Dinfo["data_model"] = df.dtypes
            Dinfo["data_drift"] = int(dP) if dP <= 100 else int(100)
            Dinfo["timestamp"] = datetime.datetime.now(tz=datetime.timezone.utc)

            Dinfo["vtag"] = str(int(v_major_ps) + int(not ps_version_inf["data_model"].equals(df.dtypes))) + "." + str(int(dP if dP <= 100 else 100)) + ".<" + Dinfo["timestamp"].strftime("%m/%d/%Y - %H:%M:%S") + ">"

        elif drift_dist == "dE":
            df = dfold[ps_version_inf["vbles_in"]]
            Dinfo = vparams_pca(df, ps_pcamodel = ps_version_inf, steps_list = steps_list)
            dE = av_gen.e_distance(Dinfo["mse"], ps_version_inf, vparams_pca, ps_version_inf["mse_model"])[0]
            
            Dinfo["data_model"] = df.dtypes
            Dinfo["data_drift"] = int(dE) if dE <= 100 else int(100)
            Dinfo["timestamp"] = datetime.datetime.now(tz=datetime.timezone.utc)

            Dinfo["vtag"] = str(int(v_major_ps) + int(not ps_version_inf["data_model"].equals(df.dtypes))) + "." + str(int(dE if dE <= 100 else 100)) + ".<" + Dinfo["timestamp"].strftime("%m/%d/%Y - %H:%M:%S") + ">"
            
        else:
            Dinfo = None

        return(Dinfo)
    
def pcamodel(X, vthreshold = 0.95, blocks = None):
    """
    Function to perform PCA on the given dataset.

    Args:
        X (DataFrame): Input dataset.
        vthreshold (float, optional): Variance threshold for selecting the number of components. Defaults to 0.95.

    Returns:
        dict: Dictionary containing the PCA model, loadings, scaler, and explained variance.
    """
    if blocks is None:
        blocks = pd.Series([*range(0, np.shape(X)[1], 1)])
    else:
        blocks = pd.Series(blocks)
    # Fit PCA

    # First, drop also variables with null variance
    X = X.loc[:,X.std() > 0]

    prepro = {"centering": np.mean(X, axis=0), "scaler": pd.Series(dict.fromkeys(blocks.index))}
    xcent = X - prepro["centering"]
    blocks_in = blocks.loc[np.intersect1d(list(xcent.columns), list(blocks.index))]
    for block_id in list(blocks_in.unique()):
        vbles_block = list(blocks_in.loc[blocks_in==block_id].index)
        prepro["scaler"][vbles_block] = xcent[vbles_block].stack().std()
    
    prepro["scaler"].dropna(inplace=True)
    xsc = xcent / prepro["scaler"]
    X = X.drop(columns=xsc.columns[xsc.isna().any()]).copy()
    prepro["centering"].drop(index = xsc.columns[xsc.isna().any()], inplace = True)
    xsc = xsc.drop(columns = xsc.columns[xsc.isna().any()]).copy()

    pca = PCA()
    pca.fit(xsc)
    ncomp = np.min(np.where(np.cumsum(pca.explained_variance_ratio_) > vthreshold))
    pca = PCA(n_components = ncomp, random_state = 42)
    pc_names = ["PC" + str(pc) for pc in range(1,pca.n_components + 1)]
    pca.fit(xsc)
    loadings = pd.DataFrame(pca.components_.T, columns = pc_names, index = X.columns.values)

    return({"model": pca, "loadings": loadings, "xscaler": prepro, "lambdaA": pca.explained_variance_})

def mfa_model(X, categorical_vars=None, num_threshold=0.95, cat_threshold=0.95):
    """
    Function to perform Multivariate Factor Analysis (MFA) on the given dataset.

    Args:
        X (DataFrame): Input dataset.
        categorical_vars (list, optional): List of column names representing categorical variables.
        num_threshold (float, optional): Variance threshold for selecting the number of components for numerical variables. Defaults to 0.95.
        cat_threshold (float, optional): Variance threshold for selecting the number of components for categorical variables. Defaults to 0.95.

    Returns:
        dict: Dictionary containing the MFA model, loadings, scaler, and explained variance.
    """

    # Separate numerical and categorical variables
    if categorical_vars is None:
        num_vars = X.select_dtypes(include=np.number)
        cat_vars = X.select_dtypes(exclude=np.number)
    else:
        num_vars = X.drop(columns=categorical_vars)
        cat_vars = X[categorical_vars]

    # Fit PCA for numerical variables
    num_scaler = StandardScaler().fit(num_vars)
    pca_num = PCA()
    pca_num.fit(num_scaler.transform(num_vars))
    num_components = np.min(np.where(np.cumsum(pca_num.explained_variance_ratio_) > num_threshold))
    pca_num = PCA(n_components=num_components)
    pca_num.fit(num_scaler.transform(num_vars))
    num_loadings = pd.DataFrame(pca_num.components_.T, columns=[f"PC{pc}" for pc in range(1, num_components + 1)], index=num_vars.columns.values)

    # Fit MFA for categorical variables
    if not cat_vars.empty:
        cat_scaler = StandardScaler().fit(cat_vars)
        fa_cat = FactorAnalysis()
        fa_cat.fit(cat_scaler.transform(cat_vars))
        cat_components = np.min(np.where(np.cumsum(fa_cat.noise_variance_) > cat_threshold))
        fa_cat = FactorAnalysis(n_components=cat_components)
        fa_cat.fit(cat_scaler.transform(cat_vars))
        cat_loadings = pd.DataFrame(fa_cat.components_.T, columns=[f"Factor{factor}" for factor in range(1, cat_components + 1)], index=cat_vars.columns.values)
    else:
        fa_cat = None
        cat_loadings = pd.DataFrame()

    return {
        "num_model": pca_num,
        "num_loadings": num_loadings,
        "num_scaler": num_scaler,
        "num_lambdaA": pca_num.explained_variance_,
        "cat_model": fa_cat,
        "cat_loadings": cat_loadings
    }

def vparams_pca(df, ps_pcamodel = None, tr_val_split = 0.8, steps_list = [5, 10, 15, 30, 50], blocks = None):
    """
    Function to compute the versioning parameters for PCA.

    Args:
        df (DataFrame): Input dataset.
        ps_pcamodel (dict, optional): PS PCA model information. Defaults to None.
        tr_val_split (float, optional): Train-validation split ratio. Defaults to 0.8.
        steps_list (list): List with the number of time steps for one-dimensional time-series data. Defaults to [5, 10, 15, 30, 50].
        blocks (dict): List with the block of variables to which each column in df pertains. Defaults to None.

    Returns:
        dict: Dictionary containing the versioning parameters for PCA.
    """
    np.random.seed(42)

    # Select only numeric variables
    if blocks is None:
        blocks = pd.Series([*range(0, np.shape(df)[1], 1)])

    dfnum = df.select_dtypes(include=np.number)
    if ps_pcamodel is None:
        if np.shape(dfnum)[1] == 1:
            mse_list = []
            # Split data into train and test sets
            train_data, test_data = train_test_split(dfnum, train_size = tr_val_split, shuffle = False, random_state=42)
            for num_steps in steps_list:
                train_data_X, __ = av_gen.create_dataset(train_data, num_steps)
                test_data_X, __ = av_gen.create_dataset(test_data, num_steps)
                trx = pd.DataFrame(train_data_X, columns=["step" + str(x) for x in range(0, num_steps)])
                tsx = pd.DataFrame(test_data_X, columns=["step" + str(x) for x in range(0, num_steps)])

                pca = pcamodel(trx, vthreshold = 0.95)
                xhat = pca["model"].inverse_transform((tsx - pca["xscaler"]["centering"]) / pca["xscaler"]["scaler"])
                rec_error = (tsx - pca["xscaler"]["centering"]) / pca["xscaler"]["scaler"] - xhat
                rss = np.sum(np.square(rec_error.stack()))
                mse = np.mean(np.square(rec_error))
                mse_list.append(mse)

            best_steps = steps_list[np.where(mse_list == min(mse_list))[0][0]]
            train_data_X, _ = av_gen.create_dataset(train_data, best_steps)
            test_data_X, _ = av_gen.create_dataset(test_data, best_steps)
            trx = pd.DataFrame(train_data_X, columns=["step" + str(x) for x in range(0, best_steps)])
            tsx = pd.DataFrame(test_data_X, columns=["step" + str(x) for x in range(0, best_steps)])

            pca = pcamodel(trx, vthreshold = 0.95, blocks = blocks)
            xhat =  pd.DataFrame(pca["model"].inverse_transform(pca["model"].transform((tsx - pca["xscaler"]["centering"]) / pca["xscaler"]["scaler"])), 
                                 columns = tsx.columns, index = tsx.index)
            rec_error = (tsx - pca["xscaler"]["centering"]) / pca["xscaler"]["scaler"] - xhat
            r2score = r2_score(((tsx - pca["xscaler"]["centering"]) / pca["xscaler"]["scaler"]).stack(), xhat.stack())
            if r2score<0:
                r2score = 1 - np.sum(np.square(rec_error.stack()))/np.sum(np.square(((tsx - pca["xscaler"]["centering"]) / pca["xscaler"]["scaler"] - pca["model"].mean_)).stack())
            rss = np.sum(np.square(rec_error.stack()))
            mse = np.mean(np.square(rec_error.stack()))
            vble_names = dfnum.columns.values
        
        else:
            
            train_data, test_data = train_test_split(dfnum, train_size = tr_val_split, shuffle = False, random_state=42)
            pca = pcamodel(train_data, blocks = blocks)
            test_data = test_data[pca["model"].feature_names_in_]
            test_sc = pd.DataFrame((test_data - pca["xscaler"]["centering"]) / pca["xscaler"]["scaler"],
                                   columns = test_data.columns, index = test_data.index)
            xhat = pd.DataFrame(pca["model"].inverse_transform(pca["model"].transform(test_sc)), columns = test_sc.columns, index = test_sc.index)
            rec_error = test_sc - xhat
            r2score = r2_score(test_sc.stack(), xhat.stack())
            if r2score<0:
                r2score = 1 - np.sum(np.square(rec_error.stack()))/np.sum(np.square(test_sc.stack()))
            rss = np.sum(np.square(rec_error.stack()))
            mse = np.mean(np.square(rec_error.stack()))
            vble_names = pca["model"].feature_names_in_

        vmodel = {"pcamodel": pca, "vbles_in": vble_names, "rss": rss, "mse": mse, "model_class": "pca", "r2score": r2score}
    else:

        pca = ps_pcamodel["pcamodel"]
        if np.shape(dfnum)[1] == 1:
            dfnum_steps, _ = av_gen.create_dataset(dfnum, len(ps_pcamodel["vbles_in"]))
            dfnum = pd.DataFrame(dfnum_steps, columns = ps_pcamodel["vbles_in"])
        if np.all([x in dfnum.columns for x in ps_pcamodel["vbles_in"]]):
            vbles_der_in = ps_pcamodel["vbles_in"]
        else:
            vbles_der_in = ps_pcamodel["vbles_in"][list(x in dfnum.columns for x in ps_pcamodel["vbles_in"])]
        dfnum_e = pd.DataFrame((dfnum[vbles_der_in] - pca["xscaler"]["centering"]) / pca["xscaler"]["scaler"], columns = vbles_der_in, index = dfnum.index)
        xhat = pd.DataFrame(pca["model"].inverse_transform(pca["model"].transform(dfnum_e)), columns = vbles_der_in, index = dfnum_e.index)
        rec_error = dfnum_e - xhat
        r2score = r2_score(dfnum_e.stack(), xhat.stack())
        rss = np.sum(np.square(rec_error.stack()))
        mse = np.mean(np.square(rec_error.stack()))
        
        vmodel = {"pcamodel": ps_pcamodel, "rss": rss, "mse": mse, "n": np.shape(dfnum)[0], "model_class" : "pca"}
    return(vmodel)

def s_distance(dfnew, ps_version_inf, alpha=0.05,  Rinfo=None):
    """
    Function to compute the distance between two datasets based on PCA loadings.

    Args:
        dfnew (DataFrame): New dataset.
        ps_version_inf (dict): PS version information.
        alpha (float, optional): Significance level for confidence intervals. Defaults to 0.05.
        Rinfo (dict, optional): DER version information. Defaults to None.

    Returns:
        dict: Dictionary containing the data drift information.
    """
    
    if Rinfo is None: Rinfo = vparams_pca(dfnew, ps_pcamodel=ps_version_inf)
    common_vbles= list(ps_version_inf["pcamodel"]["loadings"].index.intersection(Rinfo["pcamodel"]["loadings"].index))
    common_pcs = list(ps_version_inf["pcamodel"]["loadings"].columns.intersection(Rinfo["pcamodel"]["loadings"].columns))
    pc_weights = pd.Series(ps_version_inf["pcamodel"]["model"].explained_variance_ratio_, index=list(ps_version_inf["pcamodel"]["loadings"].columns.values)).loc[common_pcs]
    
    P_ps = ps_version_inf["pcamodel"]["loadings"].loc[common_vbles, common_pcs]
    P_rev = Rinfo["pcamodel"]["loadings"].loc[common_vbles, common_pcs]
    
    p_cos = [np.abs(np.dot(P_ps[a], P_rev[a])*pc_weights[a]) for a in common_pcs]
    dP = np.round(100 * (1 - sum(p_cos)))
    vinfo = {"data_drift": dP}
    return(vinfo)
