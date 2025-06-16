"""
Script Name: autoversion_gen.py
Description: this module contains various utility functions related to data manipulation
and processing which are common for both the PCA and the AE based versioning modules.

Author:
- Name: Alba González-Cebrián, Fanny Rivera-Ortiz, Jorge Mario Cortés-Mendoza, Adriana E. Chis, Michael Bradford, Horacio González-Vélez
- Email: Alba.Gonzalez-Cebrian@ncirl.ie

License: MIT License
- License URL: https://opensource.org/license/mit/
"""

import json
import ast
import os
import pickle
import random
import re
import scipy.interpolate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
 
def load_df(fname, hname = None, usecols = None, skiprows=0, dtype = float, dopickle = False, delimiter = None, decimal = ".", dojson = False,  respath= ""):
    """
    Load a dataframe from a file.
    
    Args:
        fname (str): Name of the file to load the dataframe from.
        hname (str or None): Name of the header file or header itself. If None, no header is used.
        usecols (list or None): List of column indices to load. If None, all columns are loaded.
        skiprows (int): Number of rows to skip at the beginning of the file.
        dtype (data type): Data type to use for the loaded dataframe.
        dopickle (bool): Whether to pickle the loaded dataframe.
        delimiter (str or None): Delimiter used in the file. If None, delimiter is determined based on the file extension.
        decimal (str): Decimal separator used in the file.
        dojson (bool): Whether to save the loaded dataframe as a JSON file.
        respath (str): Path to save the pickled or JSON files.
    
    Returns:
        data (pandas.DataFrame): Loaded dataframe.
    """

    fmt = os.path.splitext(fname)[1]
    file = os.path.splitext(fname)[0]
    if hname == "r":
        f = open(fname, 'r')
        df_header = f.readline().split(delimiter)
    elif hname is None :
        df_header = None
    else:
        df_header = hname
        
    if fmt == ".txt":
        data = pd.DataFrame(np.loadtxt(fname, skiprows = skiprows, dtype = dtype, usecols = usecols), columns = df_header)

    elif fmt == ".csv":
        if delimiter is None: delimiter = ","
        data = pd.read_csv(fname, skiprows = skiprows, delimiter= delimiter, decimal = decimal, names=df_header)

    elif fmt == ".json":
        data = pd.read_json(fname)
        data[data.select_dtypes(object).columns.values] = data.select_dtypes(object).apply(lambda x: x.str.replace(decimal,'.'))

    if dopickle:
        pickle.dump(data, open(respath + file + '_data.pickle', 'wb'))
    if dojson:
        json.dump(json.dumps(data), open(respath + file + '_data.json', 'w', encoding='utf-8'), ensure_ascii=False)
    else:
        return(data)
    
def dict2json(Xv, model_type):
    """
    Convert a dictionary to JSON format.
    
    Args:
        Xv (dict): Dictionary to convert.
        model_type (str): Type of the model.
    
    Returns:
        Xvjson (dict): Dictionary in JSON format.
    """

    Xvcopy = Xv.copy()

    if model_type == "ae":
        if 'model' in Xvcopy.keys(): del Xvcopy['model']
        if 'scaler' in Xvcopy.keys(): del Xvcopy['scaler']
        stypes = []
        for v in Xvcopy.values(): 
            stypes.append(str(type(v)))
        Xvcopy["keyclass"] = str(stypes)
        Xvjson = Xvcopy.copy()
        for k,v in Xvcopy.items():
            if isinstance(v, np.ndarray):
                if v.ndim == 1: Xvjson[k] = json.dumps(v.tolist())
                else: Xvjson[k] = pd.DataFrame(v).to_json(orient = 'columns')
            elif isinstance(v, (pd.DataFrame, pd.Series)): Xvjson[k] = v.to_json(orient = 'columns')
            elif isinstance(v, list): Xvjson[k] = json.dumps(v)

    elif model_type == "pca":
        Xvjson = Xv.copy()
        for k, v in Xv.items():
            if isinstance(v, np.ndarray):
                if v.ndim == 1: Xvjson[k] = v.tolist()
                else: Xvjson[k] = v.tolist()
            elif isinstance(v, (pd.DataFrame, pd.Series)): Xvjson[k] = np.array(v).tolist()
            elif isinstance(v, list):Xvjson[k] = v
    else:
        Xvjson = Xv.copy()

    return(Xvjson)

def detect_categorical_variables(df, threshold = 10):
    """
    Detect categorical variables in a Pandas DataFrame.

    Args:
        df (pandas.DataFrame): Input DataFrame to detect categorical variables.

    Returns:
        list: A list of column names representing the detected categorical variables.
    """
    categorical_vars = []
    for column in df.columns:
        unique_values = df[column].nunique()
        if isinstance(df[column].iloc[0], str):
            categorical_vars.append(column)
        elif unique_values <= threshold and pd.api.types.is_numeric_dtype(df[column].iloc[0]):
            categorical_vars.append(column)
        elif not pd.api.types.is_numeric_dtype(df[column].iloc[0]):
            categorical_vars.append(column)
    return categorical_vars

def detect_int_variables(df):
    """
    Detect categorical variables in a Pandas DataFrame.

    Args:
        df (pandas.DataFrame): Input DataFrame to detect categorical variables.

    Returns:
        list: A list of column names representing the detected categorical variables.
    """
    int_vars = []
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            if type(df[column].iloc[0]) is int:
                int_vars.append(column)
            elif np.all(df[column]>=0) and np.sum(np.abs(np.round(df[column]) - df[column])) == 0:
                int_vars.append(column)
    return int_vars

def one_hot_encode_categorical(df, categorical_vars=None, drop_original=True, enc = None):
    """
    Perform one-hot encoding of categorical variables in a Pandas DataFrame.

    Args:
        df (pandas.DataFrame): Input dataset.
        categorical_vars (list, optional): List of column names representing categorical variables. If None, categorical
                                          variables will be automatically detected. Defaults to None.
        drop_original (bool, optional): Whether to drop the original categorical columns after encoding. Defaults to True.

    Returns:
        pandas.DataFrame: DataFrame with one-hot encoded categorical variables.
    """

    # Detect categorical variables if not provided
    if categorical_vars is None or len(categorical_vars)==0:
        categorical_vars = list(df.select_dtypes(include=['object', 'category']).columns)
        if len(categorical_vars) > 0:
            # Perform one-hot encoding
            if enc == None:
                enc = OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown='ignore').fit(df[categorical_vars]).set_output(transform="pandas")  
            df_onehot = pd.concat([df, enc.transform(df[enc.feature_names_in_])], axis=1)
        else:
            enc = None
            df_onehot = df.copy()
    else:
        enc = None
        df_onehot = df.copy()


    # Drop original categorical columns if specified
    if drop_original:
        df_encoded = df_onehot.drop(columns=categorical_vars)
    else:
        df_encoded = df_onehot.copy()
    
    names_cod = list(df_encoded.columns)
    
    # Initialize an empty dictionary to store matching strings
    block_sizes = dict.fromkeys(list(df.columns))
    block_vector = dict.fromkeys(names_cod)

    # Iterate through the list of strings and compiled patterns
    
    for i, col in enumerate(list(df.columns)):
        count = 0
        pattern = re.compile(col)
        for string in names_cod:
            if pattern.search(string):
                block_vector[string] = i
                count = count+1
        block_sizes[col] = count

    block_vector = pd.Series(block_vector, index = names_cod)
    return df_encoded, enc, block_sizes, block_vector

def reverse_one_hot_encode(encoded_df, original_df, categorical_vars=None):
    """
    Reverse the one-hot encoding to convert encoded DataFrame back to original categorical variables.

    Args:
        encoded_df (pandas.DataFrame): DataFrame with one-hot encoded categorical variables.
        original_df (pandas.DataFrame): Original dataset before one-hot encoding.
        categorical_vars (list, optional): List of column names representing categorical variables. If None, categorical
                                          variables will be automatically detected. Defaults to None.

    Returns:
        pandas.DataFrame: DataFrame with original categorical variables.
    """

    # Detect categorical variables if not provided
    if categorical_vars is None:
        categorical_vars = list(original_df.select_dtypes(include=['object', 'category']).columns)

    # Create a list of column prefixes for one-hot encoding
    column_prefixes = [var + '_' for var in categorical_vars]

    # Filter columns in the encoded DataFrame that match the prefixes
    matching_columns = [col for col in encoded_df.columns if any(col.startswith(prefix) for prefix in column_prefixes)]

    # Extract the encoded columns and merge them with the original DataFrame
    encoded_data = encoded_df[matching_columns]
    original_data = original_df.drop(columns=categorical_vars)
    reversed_df = pd.concat([original_data, encoded_data], axis=1)

    return reversed_df

def block_scaling(matrix, block_sizes):
    """
    Perform block scaling on a matrix based on block sizes.

    Args:
        matrix (numpy.ndarray or pandas.DataFrame): Input matrix to be scaled.
        block_sizes (list): List of block sizes representing the number of columns in each block.

    Returns:
        numpy.ndarray or pandas.DataFrame: Scaled matrix with blocks.
    """
    if isinstance(matrix, pd.DataFrame):
        is_dataframe = True
    elif isinstance(matrix, np.ndarray):
        is_dataframe = False
    else:
        raise ValueError("Input matrix must be either a pandas DataFrame or a numpy array.")

    if len(block_sizes) == 0:
        raise ValueError("Block sizes list cannot be empty.")

    total_cols = sum(block_sizes)
    if total_cols != matrix.shape[1]:
        raise ValueError("The sum of block sizes must equal the number of columns in the matrix.")

    scaled_matrix = matrix.copy()

    if is_dataframe:
        start_col = 0
        for block_size in block_sizes:
            end_col = start_col + block_size
            block = scaled_matrix.iloc[:, start_col:end_col]
            block_mean = block.mean(axis=1)
            block_std = block.std(axis=1, ddof=0)
            block_scaled = (block - block_mean.values.reshape(-1, 1)) / block_std.values.reshape(-1, 1)
            scaled_matrix.iloc[:, start_col:end_col] = block_scaled
            start_col = end_col
    else:
        start_col = 0
        for block_size in block_sizes:
            end_col = start_col + block_size
            block = scaled_matrix[:, start_col:end_col]
            block_mean = np.mean(block, axis=1)
            block_std = np.std(block, axis=1, ddof=0)
            block_scaled = (block - block_mean.reshape(-1, 1)) / block_std.reshape(-1, 1)
            scaled_matrix[:, start_col:end_col] = block_scaled
            start_col = end_col

    return scaled_matrix


def json2dict(Xvjson, model_type, verbose = 0):
    """
    Convert a dictionary in JSON format to a regular dictionary.
    
    Args:
        Xvjson (dict): Dictionary in JSON format.
        model_type (str): Type of the model.
        verbose (int): Verbosity level.
    
    Returns:
        Xv (dict): Regular dictionary.
    """

    Xvblah = json.loads(Xvjson)
    Xv = Xvblah.copy()
    if model_type == "ae":
        stypes = dict(zip(Xvblah.keys(), Xvblah["keyclass"].split(",")))
        del stypes['vbles_in']
        for k in stypes.items():
            v = Xvblah[k]
            if "DataFrame" in k: Xv[k] = pd.DataFrame.from_dict(ast.literal_eval(v))
            elif "Series" in k: Xv[k] = pd.Series(ast.literal_eval(v))
            elif "numpy.ndarray" in k:
                if "[" in v: Xv[k] = np.array([float(x) for x in v.replace("[","").replace("]","").split(",")])
                elif "{" in v: Xv[k] = pd.DataFrame.from_dict(ast.literal_eval(v)).to_numpy()
            elif "\'int\'" in k: Xv[k] = int(v)
            elif "list" in k: Xv[k] = [float(x) for x in v.split(" ")]
            elif "\'str\'" in k: Xv[k] = v
            if verbose == 1:
                print("Converting " + k + " to " + stypes[k] + "--> " + str(type(Xv[k])))
    elif model_type == "pca":
        for k in Xvblah.keys():
            v = Xvblah[k]
            if np.ndim(v) == 2: Xv[k] = pd.DataFrame(v)
            elif np.ndim(v) == 1: Xv[k] = pd.Series(v)
        colnames = ["PC" + str(x) for x in range(1, np.shape(Xv["P"])[1]+1)]
        Xv["P"].index = Xv["vbles_in"]
        Xv["Tscores"].index = Xv["vbles_in"]
        Xv["m"].index = Xv["vbles_in"]
        Xv["s"].index = Xv["vbles_in"]
        Xv["Tscores"].columns = colnames
        Xv["lambda_A"].index = colnames
        Xv["P"].columns = colnames

    Xv["vbles_in"] =  np.asarray([str(x.replace("'","")) for x in Xvblah["vbles_in"].replace("[","").replace("]","").split(" ")])
    return(Xv)

def make_noisy_data(datadf, noise_factor = 0.5, nreps = 2, dstitle="", num_plots = 5, figs_path = "", silent = False, silentfig = True):
    """
    Generate noisy data based on the original dataset.
    
    Args:
        datadf (pandas.DataFrame): Original dataset.
        noise_factor (float): Factor to control the amount of noise.
        nreps (int): Number of noisy datasets to generate.
        dstitle (str): Title for the plot.
        num_plots (int or str): Number of plots to generate. If 'all', all plots are generated.
        figs_path (str): Path to save the generated plots.
    
    Returns:
        noisy_datafpd (pandas.DataFrame): Noisy dataset.
        denoisy_datafpd (pandas.DataFrame): Denoised dataset.
    """
    # Initialize seeds for results reproducibility
    #random.seed(42)
    np.random.seed(42) 

    # Before adding noise, scale the data for autoencoder (generally recommended)
    data_sc_np = np.asarray(datadf).astype('float32')
    # Generate noisy data set for training
    noisy_sc = data_sc_np.copy()
    # Create target dataset - i.e., multiple copies of original de-noised data 
    denoisy_sc = data_sc_np.copy()

    for irep in range(nreps):
        noisy_sc = np.append(noisy_sc, data_sc_np + noise_factor * np.random.normal(size=data_sc_np.shape), axis=0)
        denoisy_sc = np.append(denoisy_sc, data_sc_np, axis=0)
    
    noisy_datafpd = pd.DataFrame(noisy_sc, columns=datadf.columns.values)
    denoisy_datafpd = pd.DataFrame(denoisy_sc, columns=datadf.columns.values)

     # Plot original + noisy signals
    if num_plots == "all": num_plots = np.shape(datadf)[1]
    Xor_sc = pd.DataFrame(datadf, columns = datadf.columns.values, index = datadf.index.values).iloc[:, 0:min([num_plots,np.shape(datadf)[1]])]
    try:
        ax = Xor_sc.plot(subplots = True, grid = True, title = list(Xor_sc.columns.values), kind='line', label="original", color = "blue", legend=False, fontsize=9)
        for iax in range(len(ax)):
            ax[iax].plot(Xor_sc.index.values, noisy_datafpd.iloc[len(data_sc_np):2*len(data_sc_np), iax], label = "noisy " + str(noise_factor) + "$\sigma$", linestyle="--", 
                            alpha = 0.5, color = "gray", linewidth=1.5)
            ax[iax].plot(Xor_sc.index.values, noisy_datafpd.iloc[2*len(data_sc_np):3*len(data_sc_np), iax], label = "noisy " + str(noise_factor) + "$\sigma$", linestyle="--", 
                            alpha = 0.5, color = "gray", linewidth=1.5)
        idx = np.round(np.linspace(0, len(Xor_sc) - 1, min([20, len(Xor_sc)]))).astype(int)
        xlab_list = Xor_sc.index.values[idx]
        plt.tight_layout()
        plt.xticks(xlab_list, labels=xlab_list, **{'fontsize': 6}), 
        plt.savefig(figs_path + dstitle + 'ae-noisy-lines.png', dpi = 300)
        
        if not silentfig:
            plt.show(block=False)
            # FRO: Added the following line to remove the need to press a key
            #plt.pause(0.1)
            plt.close()
    except Exception as e: 
        print(e)  

    return(noisy_datafpd, denoisy_datafpd)

def optimize_knn_imputation(X, k_values):
    """
    Optimize the number of neighbors for KNN imputation.
    
    Args:
        X (pandas.DataFrame): Dataset with missing values.
        k_values (list): List of possible values for the number of neighbors.
    
    Returns:
        best_k (int): Best number of neighbors for KNN imputation.
    """
    # Initialize seeds for results reproducibility
    random.seed(42)
    #np.random.seed(42) 

    p_missing = np.sum(X.stack().isna())/np.size(X)
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
        
        rmse.append(np.sqrt(np.mean(np.square(Xtest_k.to_numpy() - Ximp.to_numpy()))))
    k_list = list(k_values)
    best_k = k_list[[i for i, x in enumerate(rmse == min(rmse)) if x][0]]
    return best_k

def clean_ds(df, mode = "r&c", nanvalue = np.nan, n_imp_neigh = 5, delete_null_var = True, threshold = 2, onehot = True, ohe = None, logcounts = True, 
             pseudocount = 1e-5, fixedvariables = []):
    """
    Clean the dataset by handling missing values and null variables.
    
    Args:
        df (pandas.DataFrame): Dataset to clean.
        mode (str): Cleaning mode. Options: 'r&c' (default), 'impute', 'r', 'c'.
        nanvalue (any): Value used to represent missing values.
        n_imp_neigh (int): Number of neighbors for KNN imputation.
        delete_null_var (bool): Whether to delete null variables.
        threshold (int): Maximum number of categories set as threshold to detect discrete variables.
        onehot (bool): Perform one hot encoding of categorical variables
        ohe (OneHotEncoder): Object from sklearn.preprocessing, previously fitted, to perform OHE on df
        logcounts (bool): Perform logarithmic transforamtion of integer variables
        pseudocount (float): Small constant value added to the data before taking the logarithm to avoid taking the logarithm of zero or a negative number, 
                            which would result in undefined or complex values. Default value set to 1e-5.
        fixedvariables (list): List of variables that must be kept in the dataset. Default value set to None.
    
    Returns:
        dict: Dictionary containing the cleaned dataframe and the list of numerical variables.
    """

    df = df.replace(nanvalue, np.nan)

    # Check variable names' format
    if isinstance(df.columns[0], tuple) :
        new_column_names = [' '.join(col) for col in df.columns]
        df.columns = new_column_names

    v_nullvar_list = list(df.columns.values[np.std(df[~np.isnan(df)], axis=0) == 0])
    v_cat_list = detect_categorical_variables(df, threshold = np.min([np.shape(df)[0]-1, threshold]))
    v_int_list = detect_int_variables(df)
    v_cat_list = [x for x in v_cat_list if x not in v_nullvar_list and x not in fixedvariables]
    v_int_list = [x for x in v_int_list if x not in v_nullvar_list and x not in fixedvariables]
    v_nullvar_list = [x for x in v_nullvar_list if x not in fixedvariables] 
    
    if delete_null_var:
        df.drop(columns=v_nullvar_list, inplace=True)
    
    if onehot:
        df_onehot, ohe, block_sizes, block_vector = one_hot_encode_categorical(df, categorical_vars = v_cat_list, enc = ohe)
        vnum = df_onehot.select_dtypes("number").columns
        df_num = df_onehot[vnum].copy()
    else:
        ohe = None
        df.drop(columns=v_cat_list, inplace=True)
        block_vector = pd.Series([*range(0, np.shape(df)[1], 1)], index=df.columns)
        #df_fact = df.copy()
        #for col in v_cat_list:
        #    df_fact[col] = pd.factorize(df[col], sort = True)[0] + 1
        vnum = df.select_dtypes("number").columns
        vnum = [x for x in vnum if x not in v_cat_list]
        df_num = df[vnum].copy()

    if logcounts:
        # Pseudocount to add to avoid zero values (can be adjusted)
        df_num[v_int_list] = np.log(df_num[v_int_list].copy() + pseudocount)

    df_na = df_num.mask(df_num == nanvalue, other=np.nan)
    if np.any(df_na.isna()):
        if mode == "impute":
            # Delete columns and rows full of missing values
            del_vars = list(df_na.columns.values[df_na.isna().sum(axis=0) == np.shape(df_na)[0]])
            del_vars = [x for x in del_vars if x not in fixedvariables]
            df_na.drop(columns = del_vars, inplace=True)
            del_rows = df_na.isna().sum(axis=1) == np.shape(df_na)[1]
            df_na.drop(index = df_na.index.values[del_rows], inplace=True)
            # Optimize the number of neighbors for imputation print("Best number of neighbors for imputation:", best_k)
            imp = KNNImputer(n_neighbors = n_imp_neigh, weights="uniform")
            dfclean = pd.DataFrame(imp.fit_transform(df_na), index=df_na.index.values, columns=df_na.columns.values)
        elif mode == "r&c":
            # Delete variables that are full of missing values and rows with missing values to compute the PCA model
            # Delete columns and rows full of missing values
            del_vars = list(df_na.columns.values[df_na.isna().sum(axis=0) == np.shape(df_na)[0]])
            del_vars = [x for x in del_vars if x not in fixedvariables]
            dfclean = df_na.drop(columns = del_vars)
            dfclean = dfclean[~dfclean.isnull().any(axis=1)]
        elif mode == "r":
            dfclean = df_na[~df_na.isnull().any(axis=1)]
        elif mode == "c":
            # Delete variables that are full of missing values
            del_vars = list(df_na.columns.values[df_na.isna().sum(axis=0) == np.shape(df_na)[0]])
            del_vars = [x for x in del_vars if x not in fixedvariables]
            dfclean = df_na.drop(columns = del_vars)
        else:
            dfclean = df_na.drop(index = df_na.index.values[df_na.isna().sum(axis=1) == np.shape(df_na)[1]])
            del_vars = list(dfclean.columns.values[dfclean.isna().sum(axis=0) == np.shape(dfclean)[0]])
            del_vars = [x for x in del_vars if x not in fixedvariables]
            dfclean = dfclean.drop(columns = del_vars)
    else:
        dfclean = df_na.copy()
    v_nullvar_list = list(dfclean.columns.values[np.std(dfclean, axis=0) == 0])
    v_nullvar_list = list(set(v_nullvar_list) - set(fixedvariables))
    if delete_null_var:
        df.drop(columns=v_nullvar_list, inplace=True)
    vnum = dfclean.select_dtypes("number").columns
    block_vector.loc[dfclean.columns]
    return({"dfclean": dfclean, "vnum": vnum, "blocks": block_vector, "ohe": ohe, "vbles_in_prepro": list(df.columns)})

def returndigit(df):
    """
    Convert string values in a dataframe to numeric format.
    
    Args:
        df (pandas.DataFrame): Dataframe to convert.
    
    Returns:
        df (pandas.DataFrame): Dataframe with converted values.
    """

    if any(df.apply(lambda x: x.str.contains("."))): df = pd.to_numeric(df, errors='coerce')
    elif any(df.apply(lambda x: x.str.contains(","))): df.apply(lambda x: x.str.replace(",", ".").float)
    return(df)

def create_dataset(X, num_steps):
    """
    Create a dataset for time series analysis.
    
    Args:
        X (pandas.DataFrame): Original dataset.
        num_steps (int): Number of time steps.
    
    Returns:
        Xd (pandas.DataFrame): Transformed dataset with the specified number of time steps.
        Yd (pandas.DataFrame): Target dataset.
    """

    Xs = []
    ys = []
    new_col_names = []
    index_steps = list(X.index.values)[:len(X)-num_steps]
    for t in range(num_steps): 
        for x in X.columns.values: 
            new_col_names.append(str(x) + "t" + str(t))
    for i in range(len(X) - num_steps):
        Xs.append(X.to_numpy()[i:i+num_steps].flatten())
        ys.append(X.to_numpy()[i+num_steps].flatten())
    Xs = np.squeeze(np.array(Xs))
    if np.ndim(Xs) ==1:
        if num_steps==1:
            Xd = pd.DataFrame(np.expand_dims(Xs, axis=1), columns = new_col_names, index = index_steps)
        elif num_steps>1:
            Xd = pd.DataFrame(np.expand_dims(Xs, axis=0), columns = new_col_names, index = index_steps)
    else:
        Xd = pd.DataFrame(Xs, columns = new_col_names, index = index_steps)
    
    Yd = pd.DataFrame(np.array(ys), columns = X.columns.values, index = index_steps)
    return Xd, Yd

def fit_mse_model(X, ps_version_info, ps_version_ml_func, path_logs = "", path_tuner = "", p=list(np.append([int(1),int(5)], np.linspace(10,100,10, dtype=int))), 
                  dstitle = "", mse_model_type = "splines", figs_path = "", resultsfile = "", silent = False, silentfig = True):
    """
    Fit an MSE model based on the specified version of the model.
    
    Args:
        X (pandas.DataFrame): Input dataset.
        ps_version_info (dict): Information about the version of the model.
        ps_version_ml_func (function): Function to train the model.
        path_logs (str): Path to save the logs.
        path_tuner (str): Path to save the tuner results.
        p (list): List of percentages for permutation.
        dstitle (str): Title for the plot.
        mse_model_type (str): Type of MSE model. Options: 'reg', 'splines'.
        figs_path (str): Path to save the generated plots.
        resultsfile (str): Name of the results file.
    
    Returns:
        dict: Dictionary containing the MSE model information.
    """
    # Initialize seeds for results reproducibility
    random.seed(42)
    #np.random.seed(42) 

    mse_k = [[] for x in range(10)]
    p_k = [[] for x in range(10)]
    mse_model_dict = {"model_class": mse_model_type}

    plt.figure()
    if mse_model_type == "splines":
        mse_model = []
        for nreps in range(0, 10):
            permuted_mse = []
            p_k[nreps] = p
            #rss_perm.append(aemodel["prss"])
            for p_level in p:
                Xnew = X.copy()
                for i, col in enumerate(X.columns):
                    ind_rep = random.sample(range(0, np.size(X[col])), int(max([1,p_level*0.01*np.size(X[col])])))
                    xvec = np.reshape(np.ravel(X[col].copy()), np.size(X[col]))
                    ind_rep_new = random.sample(range(0, np.size(ind_rep)), np.size(ind_rep))
                    xvec[ind_rep] = xvec[np.array(ind_rep)[ind_rep_new]].copy()
                    Xnew[col] = xvec
                if ps_version_info["model_class"] == "ae":
                    mse_p = ps_version_ml_func(Xnew, path_logs, path_tuner, ps_aemodel = ps_version_info)["mse"]
                elif ps_version_info["model_class"] == "pca":
                    mse_p = ps_version_ml_func(Xnew, ps_pcamodel = ps_version_info)["mse"]
                permuted_mse.append(mse_p)
            # Check only monotonically increasing values are used to fit the spline
            monotonically_increasing = [permuted_mse[0]]
            monotonically_increasing_p = [p[0]]
            for i in range(1, len(permuted_mse)):
                if permuted_mse[i] > monotonically_increasing[-1]:
                    monotonically_increasing.append(permuted_mse[i])
                    monotonically_increasing_p.append(p[i])
            sort_mse = np.sort(monotonically_increasing)
            sort_p = np.array(monotonically_increasing_p)[np.argsort(monotonically_increasing)]
            u_mse, indices = np.unique(sort_mse, return_index=True)
            spl = scipy.interpolate.UnivariateSpline(sort_mse[indices], sort_p[indices])
            mse_model.append(spl)
            plt.plot(monotonically_increasing, monotonically_increasing_p,  marker="o", color = "blue", alpha = 0.3)
            mse_k[nreps] = pd.Series(monotonically_increasing, index = monotonically_increasing_p)
        pmse_df = pd.concat(mse_k, axis = 1).sort_index()
        mean_pmse = pmse_df.median(axis=1).to_list()
        # mean_pmse = pd.DataFrame(mse_k, columns=monotonically_increasing_p).median(axis=0).to_list()
    
    elif mse_model_type == "reg":
        permuted_mse = []
        permuted_p = []
        mean_pmse = []
        # Perform permutations on each column of the dataset
        for p_level in p:
            prep = []
            mserep = []
            for nreps in range(10):
                Xnew = X.copy()
                prep.append(p_level)
                permuted_p.append(p_level)
                for i, col in enumerate(X.columns):
                    ind_rep = random.sample(range(0, np.size(X[col])), int(max([1,p_level*0.01*np.size(X[col])])))
                    xvec = np.reshape(np.ravel(X[col].copy()), np.size(X[col]))
                    ind_rep_new = ind_rep.copy()
                    np.random.shuffle(ind_rep_new)
                    xvec[ind_rep] = xvec[ind_rep_new].copy()
                    Xnew[col] = xvec
                if ps_version_info["model_class"] == "ae":
                        mse_p = ps_version_ml_func(Xnew, path_logs, path_tuner, ps_aemodel = ps_version_info)["mse"]
                elif ps_version_info["model_class"] == "pca":
                        mse_p = ps_version_ml_func(Xnew, ps_pcamodel = ps_version_info)["mse"]
                mserep.append(mse_p)       
                permuted_mse.append(mse_p)
            mean_pmse.append(np.median(mserep))
            plt.plot(mserep, prep, marker="o", color = "blue", alpha = 0.3)
        # Fit regression model between permuted percentages and MSE scores
        mse_model = np.poly1d(np.polyfit(permuted_mse, np.log(permuted_p), 1))
        mse_k = permuted_mse
    
    mse_model_dict["model_func"] = mse_model
    
    mse_model_dict["mse_range"] = [min(permuted_mse), max(permuted_mse)]
    # Show prediction for avge MSE from all the repetitions
    de_hat = [e_distance(x, ps_version_info, ps_version_ml_func, mse_model_dict, path_logs = "", path_tuner = "")[0] for x in mean_pmse]
    de_hat_decrease = [de_hat[-x]<de_hat[-x-1] for x in range(1, len(de_hat))]
    if np.any(de_hat_decrease):
        i_max = int(len(de_hat) - np.min(np.where(de_hat_decrease)) - 2)
    else:
        i_max = len(de_hat)-1
    i_min = np.where(de_hat == np.min(de_hat))[0][0]
    #mse_model_dict["mse_range"] = [min(permuted_mse), permuted_mse[i_max]]
    mse_model_dict["mse_range"] = [np.max(pmse_df,axis=1).iloc[i_min], np.min(pmse_df,axis=1).iloc[i_max]]
    de_hat[int(i_max):] = list(100*np.ones(len(de_hat)-i_max))
    p_true = pmse_df.index.values.tolist()
    mse_model_dict["r2_score"] = r2_score(p_true, de_hat)
    mse_sort = np.sort(mean_pmse)
    phat_sort =  np.array(de_hat)[np.argsort(mean_pmse)]
    plt.plot(mse_sort, phat_sort, color="red", linewidth=3, linestyle = "--", label="y = $a\cdot \exp(b\cdot x)$, R$^2$ = " + str(r2_score))
    plt.ylabel("Permutation level (%)")
    plt.xlabel("MSE model (" + ps_version_info["model_class"] + ", " + mse_model_type + ")")
    plt.grid(True, alpha = 0.2, which = "both")
    plt.title(dstitle)
    if os.path.isdir(figs_path):
        plt.savefig(figs_path + resultsfile + dstitle + ps_version_info["model_class"] + "-" + mse_model_type + '-permutation-error-curves-regmodel.png', dpi = 300)
    else:
        os.mkdir(figs_path)
        plt.savefig(figs_path + resultsfile + dstitle + ps_version_info["model_class"] + "-" + mse_model_type + '-permutation-error-curves-regmodel.png', dpi = 300)
    if not silentfig:
        plt.show(block=False)
        # FRO: Added the following line to remove the need to press a key
        #plt.pause(0.1)
        plt.close()
    
    return mse_model_dict

def e_distance(x_new, ps_version_info, ps_version_ml_func, ps_version_mse_func, path_logs = "", path_tuner = ""):
    """
    Calculate the Euclidean distance based on the MSE model.
    
    Args:
        x_new (pandas.DataFrame or float): New input data.
        ps_version_info (dict): Information about the version of the model.
        ps_version_ml_func (function): Function to train the model.
        ps_version_mse_func (dict): MSE model information.
        path_logs (str): Path to save the logs.
        path_tuner (str): Path to save the tuner results.
    
    Returns:
        list: List containing the predicted permutation level and the MSE value.
    """
    
    if isinstance(x_new, pd.DataFrame): 
        if np.all([x in x_new.columns for x in ps_version_info["vbles_in"]]):
            vbles_der_in = ps_version_info["vbles_in"]
        else:
            vbles_der_in = ps_version_info["vbles_in"][list(x in x_new.columns for x in ps_version_info["vbles_in"])]      
        x_new_in = x_new[vbles_der_in]
        if ps_version_info["model_class"] == "ae": 
            mse_x = ps_version_ml_func(x_new_in, path_logs, path_tuner, ps_model = ps_version_info)["mse"]
        elif ps_version_info["model_class"] == "pca": 
            mse_x = ps_version_ml_func(x_new_in, ps_pcamodel = ps_version_info)["mse"]    
    else: mse_x = x_new
    # Check that the new MSE value is within the limits used to fit the MSE models:
    if mse_x < ps_version_mse_func["mse_range"][0]: 
        p_hat = 0 # If it's below, dE = 0
    elif mse_x > ps_version_mse_func["mse_range"][1]: 
        p_hat = 100 # If it's above, dE = 100
    else:
        if ps_version_mse_func["model_class"] == "splines":
            p_k = []
            for spl_k in ps_version_mse_func["model_func"]: 
                phat_k = spl_k(mse_x)
                if phat_k<=0: phat_k = 0
                elif phat_k>=100: phat_k = 100
                p_k.append(phat_k)
            p_hat = np.median(p_k)
        elif ps_version_mse_func["model_class"] == "reg":
            p_hat = np.exp(ps_version_mse_func["model_func"](mse_x))
        if p_hat < 0: p_hat = 0
        elif p_hat > 100: p_hat = 100
    return [p_hat, mse_x]
