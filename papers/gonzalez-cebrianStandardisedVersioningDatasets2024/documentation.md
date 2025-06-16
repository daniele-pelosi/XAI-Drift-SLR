# General description
Data versioning can be used to track changes and assess the drift between datasets. The applications of such a tool range from data quality control and monitoring in various applications, to better management of data resources in repositories. 

The code provided in this GitHub repository is structured in four main python modules or files:

1. `sim_experiments.py`: this module provides the set of functions to assist in a variety of tasks related to the experimentation with the time series datasets. This toolkit enables you to handle tasks like exporting performance metrics to Excel files, executing Primary Source (PS) versioning, initializing dictionaries for storing the versioning results across multiple repetitions, comparing and filtering variables across datasets, calculating versioning metrics, and more. In addition, you can use this module to modify your data emulating the creation (C), update (U) and deletion (D) experiments. Furthermore, it can help you extract seasonal and trend components from time series data and visualize these components. In a technical sense, `sim_experiments.py` serves for the execution of a the CUD experiments and analyses related to time series data.
2. `autoversion_service.py`: this module is a Python script designed to perform PCA-based versioning with data drift detection using two different approaches: PS (Primary Source) and DER (Derived). It defines several functions and includes options for loading datasets, cleaning data, performing Principal Component Analysis (PCA), and computing versioning parameters. 
3. `autoversion_aemod.py`: this module provides the functionality to apply versioning using autoencoders (AE) to compute the data drift term. To do so, it includes the necessary functions to perform model training and compute the version attributes. 
4. `autoversion_gen.py`: this module contains various utility functions related to data manipulation and processing which are common for both the PCA and the AE based versioning modules. 

The following sections give further information about each one of the modules.
# `sim_experiments.py`
### `exp_write_results`:
   - Writes performance metrics to an Excel file.
   - Input:
     - `perf_metrics`: Dictionary containing performance metrics.
     - `path_resul`: Path to the results file.
     - `do_pca` (optional): Boolean, whether to include PCA metrics (default: True).
     - `do_ae` (optional): Boolean, whether to include AE metrics (default: True).
   
### `exp_PSinfo`:
   - Performs Primary Source (PS) versioning and records performance metrics.
   - Input:
     - `dataPS`: Primary Source data in a Pandas DataFrame.
     - `PS_dic`: Dictionary to store the PS versions and metrics.
     - `v_time`: Dictionary to store the versioning time for each metric.
     - `resultspath` (optional): Path to the results directory (default: "").
     - `mod_pca` (optional): Boolean, whether to perform PCA versioning (default: True).
     - `mod_ae` (optional): Boolean, whether to perform AE versioning (default: False).
     - `naval` (optional): NaN value to use for imputation (default: np.nan).
     - `dstitle` (optional): Title for the dataset (default: "").
     - `mse_model_type` (optional): MSE model type (default: "reg").
     - `tr_val_split` (optional): Train/Validation split ratio (default: 0.8).
     - `noisefactor` (optional): Noise factor for AE versioning (default: 0.05).
     - `yname` (optional): Name of the target variable (default: None).
     - `resultsfile` (optional): Name of the results file (default: "").

### `init_dicts`:
   - Initializes dictionaries to store versioning results.
   - Input:
     - `mod_pca`: Boolean, whether PCA versioning is enabled.
     - `mod_ae`: Boolean, whether AE versioning is enabled.
   - Returns a tuple of initialized dictionaries.

### `get_PS_data`:
   - Performs Primary Source (PS) data processing and versioning.
   - Input:
     - `dataset`: Input dataset in a Pandas DataFrame.
     - `resultspath` (optional): Path to the results directory (default: "").
     - `resultsfile` (optional): Name of the results file (default: "").
     - `mod_pca` (optional): Boolean, whether to perform PCA versioning (default: True).
     - `mod_ae` (optional): Boolean, whether to perform AE versioning (default: True).
     - `naval` (optional): NaN value to use for imputation (default: np.nan).
     - `dstitle` (optional): Title for the dataset (default: "").
     - `pctge_PS` (optional): Percentage of Primary Source data (default: 0.1).
     - `yname` (optional): Name of the target variable (default: None).
     - `PSfilename` (optional): Name of the PS model file (default: "psmodels.pkl").
     - `dopickle` (optional): Whether to pickle the results (default: False).
     - `tr_val_split` (optional): Train/Validation split ratio (default: 0.75).
     - `mse_model_type` (optional): MSE model type (default: "splines").
     - `noisefactor` (optional): Noise factor for AE versioning (default: 0.05).
     - `cat_threshold` (optional): Threshold for categorical variables (default: 10).
   - Returns a tuple containing PS dictionaries and indices (if `dopickle` is True).

### `filter_commonvars`:
   - Filters out common variables between two datasets.
   - Input:
     - `X1`: First dataset in a Pandas DataFrame.
     - `X2`: Second dataset in a Pandas DataFrame.
   - Returns a tuple of filtered datasets and removed variables.

### `vmetrics_rep`:
   - Computes versioning metrics for a given dataset using repetition-based evaluation.
   - Input:
     - `X`: Input dataset in a Pandas DataFrame.
     - `pcamodPS` (optional): Dictionary containing PCA models for PS versioning (default: None).
     - `aemodPS` (optional): Dictionary containing AE models for PS versioning (default: None).
     - `Y` (optional): Target variable in a Pandas DataFrame (default: None).
     - `nrep` (optional): Number of repetitions (default: 50).
     - `naval` (optional): NaN value to use for imputation (default: np.nan).
     - `dstitle` (optional): Title for the dataset (default: "").
     - `yname` (optional): Name of the target variable (default: None).
   - Returns a dictionary of versioning metrics.

### `rem_rows_exp`:
   - Removes rows from the dataset based on the specified mode.
   - Input:
     - `dataset`: Input dataset in a Pandas DataFrame.
     - `mode` (optional): Mode of row removal (default: "rnd" - random).
     - `rm_indices` (optional): List of indices to remove when mode is "set".
     - `level_artifact` (optional): Level of artifact to introduce, determining the proportion of rows to remove (default: 0.01).
   - Returns the modified dataset with rows removed.

### `trans_cols_exp`:
   - Transforms columns of the dataset based on the specified mode.
   - Input:
     - `dataset`: Input dataset in a Pandas DataFrame.
     - `level_artifact` (optional): Level of artifact to introduce, determining the proportion of columns to transform.
     - `mode` (optional): Transformation mode (default: "cbrt" - cubic root).
   - Returns the modified dataset with transformed columns.

### `trans_rows_exp`:
   - Transforms rows of the dataset by normalizing each row.
   - Input:
     - `dataset`: Input dataset in a Pandas DataFrame.
   - Returns the modified dataset with transformed rows.

### `miss_imp_exp`:
   - Performs missing data imputation on the dataset based on the specified mode.
   - Input:
      - `dataset`: Input dataset in a Pandas DataFrame.
      - `mode` (optional): Imputation mode (default: "mv" - multivariate).
      - `level_artifact` (optional): Level of artifact to introduce, determining the proportion of missing entries to impute.
      - `imp_strat` (optional): Imputation strategy used when mode is "uv" (default: "mean").
      - `nmax_imp_iter` (optional): Maximum number of imputation iterations when mode is "mv" (default: 10).
      - `nmax_imp_neigh` (optional): Maximum number of nearest neighbors to consider when mode is "knn" (default: 10).
   - Returns the modified dataset with missing data imputed and the imputation mode used.

### `optimize_knn_imputation`:
 - Optimizes the number of nearest neighbors for KNN imputation.
 - Input:
   - `X`: Input dataset.
   - `k_values`: List of candidate values for the number of nearest neighbors.
 - Returns the optimized number of nearest neighbors.

### `nonlintrans`:
 - Performs a nonlinear transformation on the input dataset.
 - Input:
   - `X`: Input dataset.
   - `mode` (optional): Transformation mode (default: "cbrt" - cubic root).
   - `epsilon` (optional): Epsilon value used for logarithmic transformation (default: 10e-4).
 - Returns the transformed dataset.

### `extract_seasonality_trend`:
 - Extracts seasonal and trend components from a multivariate time series dataset.
 - Input:
   - `data`: Multivariate time series dataset.
   - `freq`: Frequency of the data (e.g., 7 for weekly data, 12 for monthly data).
 - Returns seasonal and trend components.

### `convert_tuples_to_dates`:
 - Converts a list of tuples into a list of dates.
 - Input:
   - `tuple_list`: List of tuples.
 - Returns a list of dates.

### `plot_time_components`:
 - Plots the original data, trend component, and seasonal component of a time series.
 - Input:
   - `dataf`: Input time series data.
   - `freq`: Frequency of the time series.
   - `xplot` (optional): X-axis values for plotting (default: None).
   - `path_figs` (optional): Path to save the generated figures (default: "figures/").
   - `dsname` (optional): Name of the dataset (default: "").
   - `silent` (optional): Whether to display the plot (default: True).

### `plot_time_components_combined`:
 - Plots the original data, trend component, and seasonal component of two combined time series.
 - Input:
   - `dataf_ps`: Primary time series data.
   - `dataf_rev`: Revised time series data.
   - `freq`: Frequency of the time series.
   - `xplot_ps` (optional): X-axis values for primary time series plotting (default: None).
   - `xplot_rev` (optional): X-axis values for revised time series plotting (default: None).
   - `path_figs` (optional): Path to save the generated figures (default: "figures/").
   - `dsname` (optional): Name of the dataset (default: "").
   - `silent` (optional): Whether to display the plot (default: True).

### `plot_time_components_div`:
 - Plots the original data, trend component, and seasonal component of a divided time series.
 - Input:
   - `dataf`: Input time series data.
   - `ind_ps`: Index values corresponding to the primary time series.
   - `ind_rev`: Index values corresponding to the revised time series.
   - `freq`: Frequency of the time series.
   - `xplot_ps` (optional): X-axis values for primary time series plotting (default: None).
   - `xplot_rev` (optional): X-axis values for revised time series plotting (default: None).
   - `path_figs` (optional): Path to save the generated figures (default: "figures/").
   - `dsname` (optional): Name of the dataset (default: "").
   - `xtxtsize` (optional): Font size of x-axis labels (default: 8).
   - `onlydate` (optional): Whether to display only the date in x-axis labels (default: True).
   - `silent` (optional): Whether to display the plot (default: True).

# `autoversion_service.py`
### `versionPS`:
- Performs PCA-based versioning using the PS approach with data drift detection.
- Input: 
  - `df` (DataFrame): Input dataset.
  - Various optional parameters to customize the process, such as data cleaning, PCA settings, and output options.
- Output: 
  - Returns version information as a dictionary or a JSON-formatted string.

### `versionDER`:
- Performs PCA-based versioning using the DER approach with data drift detection.
- Input: 
  - `df` (DataFrame): Input dataset.
  - `ps_version_inf` (dict or str): PS version information.
  - Various optional parameters for data cleaning, PCA settings, and output options.
- Output: 
  - Returns version information as a dictionary or a JSON-formatted string.

### `version_info_pca`:
- Computes version information and detects data drift using PCA for either PS or DER.
- Input: 
  - `df` (DataFrame): Input dataset.
  - Various optional parameters, including mode, significance levels, and file paths for saving results.
- Output: 
  - Returns version information as a dictionary.

### `pcamodel`:
- Performs PCA (Principal Component Analysis) on the given dataset.
- Input: 
  - `X` (DataFrame): Input dataset.
  - `vthreshold` (float, optional): Variance threshold for selecting the number of components.
- Output: 
  - Returns a dictionary containing the PCA model, loadings, scaler, and explained variance.

### `mfa_model`:
- Performs Multivariate Factor Analysis (MFA) on the given dataset, considering both numerical and categorical variables.
- Input: 
  - `X` (DataFrame): Input dataset.
  - Various optional parameters for threshold values.
- Output: 
  - Returns a dictionary containing the MFA model, loadings, scaler, and explained variance.

### `vparams_pca`:
- Computes versioning parameters for PCA.
- Input: 
  - `df` (DataFrame): Input dataset.
  - Various optional parameters, including the split ratio and the list of time steps.
- Output: 
  - Returns a dictionary containing versioning parameters.

### `s_distance`:
- Purpose: Computes the distance between two datasets based on PCA loadings.
- Input: 
  - `dfnew` (DataFrame): New dataset.
  - `ps_version_inf` (dict): PS version information.
  - Various optional parameters, including the significance level.
- Output: 
  - Returns data drift information as a dictionary.

# `autoversion_aemod.py`:
### `versionPS`:
- Apply versioning based on autoencoders to the dataset.
- Input:
  - `df` (pd.DataFrame): The input dataset.
  - `path_logs` (str): The path to store the logs.
  - `path_tuner` (str): The path for tuner.
  - `modeclean` (str, optional): The mode of cleaning. Default is "r&c".
  - `nanvalue` (float, optional): The value representing missing data. Default is np.nan.
  - `nrep` (int, optional): The number of repetitions. Default is 50.
  - `alpha` (float, optional): The alpha value. Default is 0.05.
  - `outdict` (bool, optional): Flag to output dictionary. Default is False.
  - `loadfile` (bool, optional): Flag to load the file. Default is True.
  - `hname` (str, optional): The name for the hypermodel. Default is None.
  - `usecols` (list, optional): The list of columns to use. Default is None.
  - `skiprows` (int, optional): The number of rows to skip. Default is 0.
  - `dtype` (type, optional): The data type. Default is float.
  - `dopickle` (bool, optional): Flag to pickle the data. Default is False.
  - `delimiter` (str, optional): The delimiter. Default is None.
  - `respath` (str, optional): The path for results. Default is "".
  - `decimal` (str, optional): The decimal representation. Default is ".".
  - `dojson` (bool, optional): Flag to use JSON format. Default is False.
  - `dstitle` (str, optional): The title for the dataset. Default is "".
  - `epochs_tuner` (int, optional): The number of epochs for tuning. Default is 40.
  - `epochs_fit` (int, optional): The number of epochs for fitting. Default is 20.
  - `num_plots` (int, optional): The number of plots. Default is 5.
  - `verbose` (int, optional): The verbosity level. Default is 0.
  - `mse_model_type` (str, optional): The type of MSE model. Default is "reg".
  - `tr_val_split` (float, optional): The train/validation split ratio. Default is 0.8.
  - `steps_list` (list, optional): The list with the number of time steps. Default is [5, 10, 15, 30, 50].
  - `noisefactor` (float, optional): The noise factor. Default is 0.05.
  - `figs_path` (str, optional): The path for saving figures. Default is "".
  - `resultsfile` (str, optional): The path for saving results. Default is "".
  - `cat_threshold` (int, optional): The categorical threshold. Default is 10.
  - `silent` (bool, optional): Flag for silent mode. Default is True.
- Output:
  - `dict or None`: The version information if `outdict` is True, else None.

### `versionDER`:
- Apply versioning based on autoencoders to the dataset with respect to the reference version.
- Input:
  - `df` (pd.DataFrame): The input dataset.
  - `ps_version_inf` (dict or str): The reference version information or path to the reference version information.
  - `path_logs` (str): The path to store the logs.
  - `path_tuner` (str): The path for tuner.
  - `modeclean` (str, optional): The mode of cleaning. Default is "r&c".
  - `nanvalue` (float, optional): The value representing missing data. Default is np.nan.
  - `drift_dist` (str, optional): The drift distance measure. Default is "dE".
  - `outdict` (bool, optional): Flag to output dictionary. Default is False.
  - `loadfile` (bool, optional): Flag to load file. Default is True.
  - `hname` (str, optional): The name for the hypermodel. Default is None.
  - `usecols` (list, optional): The list of columns to use. Default is None.
  - `skiprows` (int, optional): The number of rows to skip. Default is 0.
  - `dtype` (type, optional): The data type. Default is float.
  - `dopickle` (bool, optional): Flag to pickle the data. Default is False.
  - `delimiter` (str, optional): The delimiter. Default is None.
  - `respath` (str, optional): The path for results. Default is "".
  - `decimal` (str, optional): The decimal representation. Default is ".".
  - `dojson` (bool, optional): Flag to use JSON format. Default is False.
  - `dstitle` (str, optional): The title for the dataset. Default is "".
  - `epochs_tuner` (int, optional): The number of epochs for tuning. Default is 40.
  - `epochs_fit` (int, optional): The number of epochs for fitting. Default is 20.
  - `num_plots` (int, optional): The number of plots. Default is 5.
  - `cat_threshold` (int, optional): The categorical threshold. Default is 10.
  - `silent` (bool, optional): Flag for silent mode. Default is True.
- Output:
  - `dict or None`: The version information if `outdict` is True, else None.

### `version_info_ae`:
- Obtain version information based on autoencoders.
- Input:
  - `X` (pd.DataFrame): The input dataset.
  - `path_logs` (str): The path to store the logs.
  - `path_tuner` (str): The path for tuner.
  - `mode` (str, optional): The mode of versioning. Default is "ps".
  - `PSv` (dict or None, optional): The reference version information or None. Default is None.
  - `drift_dist` (str, optional): The drift distance measure. Default is "S".
  - `nrep` (int, optional): The number of repetitions. Default is 30.
  - `alpha` (float, optional): The alpha value. Default is 0.05.
  - `dstitle` (str, optional): The title for the dataset. Default is "".
  - `epochs_tuner` (int, optional): The number of epochs for tuning. Default is 40.
  - `epochs_fit` (int, optional): The number of epochs for fitting. Default is 20.
  - `cat_threshold` (int, optional): The categorical threshold. Default is 10.
  - `silent` (bool, optional): Flag for silent mode. Default is True.
- Output:
  - `dict`: The version information.

### `MyAE` Class:
- A custom Autoencoder model based on Keras's Model class.
- Constructor Arguments:
  - `hp` (Hyperparameters): Hyperparameters for the model.
  - `numPredictors` (int): Number of predictors.

### `MyAEHyperModel` Class:
- A HyperModel class for the Autoencoder.
- Constructor Arguments:
  - `numPredictors` (int): Number of predictors.

### `Getautoencoder` Function:
- Build an Autoencoder model based on hyperparameters.
- Input:
  - `hp` (Hyperparameters): Hyperparameters for the model.
  - `numPredictors` (int): Number of predictors.
- Output:
  - `tf.keras.Model`: The Autoencoder model.

### `vparams_ae` Function:
- Obtain parameters based on autoencoders.
- Input:
  - `X` (pd.DataFrame): The input dataset.
  - `path_logs` (str): The path to store the logs.
  - `path_tuner` (str): The path for tuner.
  - `ps_aemodel` (dict or None, optional): The reference version information or None. Default is None.
  - `noisetr` (bool, optional): Flag to add noise to the training data. Default is True.
  - `dstitle` (str, optional): The title for the dataset. Default is "".
  - `epochs_tuner` (int, optional): The number of epochs for tuning. Default is 40.
  - `epochs_fit` (int, optional): The number of epochs for fitting. Default is 20.
  - `num_plots` (int, optional): The number of plots. Default is 5.
  - `tr_val_split` (float, optional): The train/validation split ratio. Default is 0.8.
  - `noisefactor` (float, optional): The noise factor. Default is 0.5.
  - `steps_list` (list, optional): The list with the numbers of time steps. Default is [5, 10, 15, 30, 50].
  - `figs_path` (str, optional): The path for saving figures. Default is "".
  - `resultsfile` (str, optional): The path for saving results. Default is "".
  - `silent` (bool, optional): Flag for silent mode. Default is True.
- Output:
  - `dict`: The version information.

### `filter_commonvars` Function:
- Filters the dataset to include only common variables present in the reference list.
- Input:
  - `ref_list` (list): The reference list of variables.
  - `X` (pd.DataFrame): The input dataset.
- Output:
  - `pd.DataFrame`: The filtered dataset.
  - `list`: The removed variables.

These descriptions provide details on the input arguments and what each function or class does. If you have any specific questions about how these functions/classes work or how to use them, please feel free to ask.

# `autoversion_gen.py`:
### `load_df`:
- Load a dataframe from a file.
- Input:
  - `fname` (str): Name of the file to load the dataframe from.
  - `hname` (str or None): Name of the header file or header itself. If None, no header is used.
  - `usecols` (list or None): List of column indices to load. If None, all columns are loaded.
  - `skiprows` (int): Number of rows to skip at the beginning of the file.
  - `dtype` (data type): Data type to use for the loaded dataframe.
  - `dopickle` (bool): Whether to pickle the loaded dataframe.
  - `delimiter` (str or None): Delimiter used in the file. If None, delimiter is determined based on the file extension.
  - `decimal` (str): Decimal separator used in the file.
  - `dojson` (bool): Whether to save the loaded dataframe as a JSON file.
  - `respath` (str): Path to save the pickled or JSON files.
- Output:
  - `data` (pandas.DataFrame): Loaded dataframe.

### `dict2json`:
- Convert a dictionary to JSON format.
- Input:
  - `Xv` (dict): Dictionary to convert.
  - `model_type` (str): Type of the model.
- Output:
  - `Xvjson` (dict): Dictionary in JSON format.

### `detect_categorical_variables`:
- Detect categorical variables in a Pandas DataFrame.
- Input:
  - `df` (pandas.DataFrame): Input DataFrame to detect categorical variables.
- Output:
  - `list`: A list of column names representing the detected categorical variables.

### `detect_int_variables`:
- Detect integer variables in a Pandas DataFrame.
- Input:
  - `df` (pandas.DataFrame): Input DataFrame to detect integer variables.
- Output:
  - `list`: A list of column names representing the detected integer variables.

### `one_hot_encode_categorical`:
- Perform one-hot encoding of categorical variables in a Pandas DataFrame.
- Input:
  - `df` (pandas.DataFrame): Input dataset.
  - `categorical_vars` (list, optional): List of column names representing categorical variables. If None, categorical variables will be automatically detected. Defaults to None.
  - `drop_original` (bool, optional): Whether to drop the original categorical columns after encoding. Defaults to True.
  - `enc` (OneHotEncoder): Object from sklearn.preprocessing, previously fitted, to perform OHE on df.
- Output:
  - `pandas.DataFrame`: DataFrame with one-hot encoded categorical variables.

### `reverse_one_hot_encode`:
- Reverse the one-hot encoding to convert encoded DataFrame back to original categorical variables.
- Input:
  - `encoded_df` (pandas.DataFrame): DataFrame with one-hot encoded categorical variables.
  - `original_df` (pandas.DataFrame): Original dataset before one-hot encoding.
  - `categorical_vars` (list, optional): List of column names representing categorical variables. If None, categorical variables will be automatically detected. Defaults to None.
- Output:
  - `pandas.DataFrame`: DataFrame with original categorical variables.

### `block_scaling`:
- Perform block scaling on a matrix based on block sizes.
- Input:
  - `matrix` (numpy.ndarray or pandas.DataFrame): Input matrix to be scaled.
  - `block_sizes` (list): List of block sizes representing the number of columns in each block.
- Output:
  - `numpy.ndarray` or `pandas.DataFrame`: Scaled matrix with blocks.

### `json2dict`:
- Convert a dictionary in JSON format to a regular dictionary.
- Input:
  - `Xvjson` (dict): Dictionary in JSON format.
  - `model_type` (str): Type of the model.
  - `verbose` (int): Verbosity level.
- Output:
  - `Xv` (dict): Regular dictionary.

### `make_noisy_data`:
- Generate noisy data based on the original dataset.
- Input:
  - `datadf` (pandas.DataFrame): Original dataset.
  - `noise_factor` (float): Factor to control the amount of noise.
  - `nreps` (int): Number of noisy datasets to generate.
  - `dstitle` (str): Title for the plot.
  - `num_plots` (int or str): Number of plots to generate. If 'all', all plots are generated.
  - `figs_path` (str): Path to save the generated plots.
  - `silent` (bool): Whether to suppress display of the plot.
- Output:
  - `noisy_datafpd` (pandas.DataFrame): Noisy dataset.
  - `denoisy_datafpd` (pandas.DataFrame): Denoised dataset.

### `optimize_knn_imputation`:
- Optimize the number of neighbors for KNN imputation.
- Input:
  - `X` (pandas.DataFrame): Dataset with missing values.
  - `k_values` (list): List of possible values for the number of neighbors.
- Output:
  - `best_k` (int): Best number of neighbors for KNN imputation.

### `clean_ds`:
- Clean the dataset by handling missing values and null variables.
- Input:
  - `df` (pandas.DataFrame): Dataset to clean.
  - `mode` (str): Cleaning mode. Options: 'r&c' (default), 'impute', 'r', 'c'.
  - `nanvalue` (any): Value used to represent missing values.
  - `n_imp_neigh` (int): Number of neighbors for KNN imputation.
  - `delete_null_var` (bool): Whether to delete null variables.
  - `threshold` (int): Maximum number of categories set as threshold to detect discrete variables.
  - `onehot` (bool): Perform one hot encoding of categorical variables.
  - `ohe` (OneHotEncoder): Object from sklearn.preprocessing, previously fitted, to perform OHE on df.
  - `logcounts` (bool): Perform logarithmic transformation of integer variables.
  - `pseudocount` (float): Small constant value added to the data before taking the logarithm to avoid taking the logarithm of zero or a negative number, which would result in undefined or complex values. Default value set to 1e-5.
  - `fixedvariables` (list): List of variables that must be kept in the dataset. Default value set to None.
- Output: dictionary containing the cleaned dataframe and the list of numerical variables.

### `returndigit`:
- Convert string values in a dataframe to numeric format.
- Input:
  - `df` (pandas.DataFrame): Dataframe to convert.
- Output: dataframe with converted values.

### `create_dataset`:
- Create a dataset for time series analysis.
- Input:
  - `X` (pandas.DataFrame): Original dataset.
  - `num_steps` (int): Number of time steps.
- Output: transformed dataset with the specified number of time steps and the target dataset.

### `fit_mse_model`:
- Fit an MSE model based on the specified version of the model.
- Input:
  - `X` (pandas.DataFrame): Input dataset.
  - `ps_version_info` (dict): Information about the version of the model.
  - `ps_version_ml_func` (function): Function to train the model.
  - `path_logs` (str): Path to save the logs.
  - `path_tuner` (str): Path to save the tuner results.
  - `p` (list): List of percentages for permutation.
  - `dstitle` (str): Title for the plot.
  - `mse_model_type` (str): Type of MSE model. Options: 'reg', 'splines'.
  - `figs_path` (str): Path to save the generated plots.
  - `resultsfile` (str): Name of the results file.
  - `silent` (bool): Suppress display of the plot.
- Output: dictionary containing the MSE model information.

### `e_distance`:
- Calculate the Euclidean distance based on the MSE model.
- Input:
  - `x_new` (pandas.DataFrame or float): New input data.
  - `ps_version_info` (dict): Information about the version of the model.
  - `ps_version_ml_func` (function): Function to train the model.
  - `ps_version_mse_func` (dict): MSE model information.
  - `path_logs` (str): Path to save the logs.
  - `path_tuner` (str): Path to save the tuner results.
- Output: list containing the predicted permutation level and the MSE value.

