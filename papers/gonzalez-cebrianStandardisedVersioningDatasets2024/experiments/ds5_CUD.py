"""
Script Name: ds5_CUD.py
Description: this script executes the Creation, Update and Deletion experiments with DS 05.

Author:
- Name: Alba González-Cebrián, Adriana E. Chis, Michael Bradford, Horacio González-Vélez
- Email: Alba.Gonzalez-Cebrian@ncirl.ie

License: MIT License
- License URL: https://opensource.org/license/mit/
"""
# %% [markdown]
# # Data set 5<br>
# This dataset included responses from a gas multisensor device deployed in an Italian city from 2004 to April 2005. Hourly response averages and gas concentration references from a certified analyzer were recorded. For the experiments involving the creation of new records, the data from 2004 was used as the Primary Source, and measurements from 2005 were treated as the new records.<br>
# ## Preliminaries<br>
# Import and load the uses Python packages and modules:

# %%
import sys
import os

# Add the parent directory to sys.path
parent_directory = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                                '../main/'))
sys.path.insert(0, parent_directory)
import pandas as pd
import numpy as np
import sim_experiments as smexp_dyn
from datetime import date

# %% [markdown]
# Load the data set and prepare it:
# FRO Added message to indicate we have started to run the demo
print("Loading the Experiment Configuration ...")

# %%
df = pd.read_csv("../datasets/ds05_AirQualityUCI.csv", sep=";", decimal=",", na_values=-200)
df.drop(df.columns.values[-2:], axis=1, inplace=True)
# Create a pandas dataframe from the data
df["Date"] = pd.to_datetime(df["Date"],format = '%d/%m/%Y')
df["MonthYear"] = df["Date"].dt.to_period('M')
df["Time"] = pd.to_datetime(df["Time"],format = '%H.%M.%S').dt.time
df.set_index(["Date", "Time"], inplace=True)
dfnum = df.select_dtypes([float, int])
dfnum = dfnum.replace(-200, np.nan).iloc[0:9357,:]
dfnum.drop(columns="NMHC(GT)", inplace=True)

# %%
X_PS = dfnum.loc[df["MonthYear"]<"2005-01-01",:]
X_NEW = dfnum.loc[df["MonthYear"]>="2005-01-01",:]

# %% [markdown]
# Exploratory analysis showing time series decomposigion as an additive model where each time instant ($x_i$) is the addition of a trend component ($T_i$), a seasonal component ($S_i$) and an error component ($E_i$)

# %%
smexp_dyn.plot_time_components_div(pd.concat([X_PS, X_NEW]), X_PS.index, X_NEW.index, 24*7, dsname="ds05comb_weekly", xtxtsize=5, path_figs="../figures/")

# %% [markdown]
# ## Primary Source models<br>
# Obtain the parameters for the reference batch of data. The function returns a dictionary *ps_dict* with the parameters to compute each one of the drift metrics according to a different ML model and the _indPS_ variable with the indices of the records used for the reference set.

# %%
import random
import tensorflow as tf
tf.random.set_seed(42) 
np.random.seed(42)
random.seed(42)
print("Doing Primary Source Model -- \n" + "N = " + str(len(X_PS)) + " (" + str(np.round(len(X_PS)/(len(X_PS) + len(X_NEW))*100,2)) + " %" + " of total dataset length)")
ps_dict, ind_PS = smexp_dyn.get_PS_data(X_PS, resultspath = "../results/ds05/", dstitle = "DS 05 PS", 
                                        PSfilename = "ds05ps.pkl", mse_model_type="splines", pctge_PS=1, resultsfile="dyn-ds-05")

# %% [markdown]
# ### New versions<br>
# When a new version of the dataset is generated, it will be compared to the information from the previous version in the following way:<br>
#   * $d_{P}$: computes the cosine distance between loading matrices obtained for both data sets;<br>
#   * $d_{E, PCA}$: computes the MSE obtained by reconstructing the new batch using the reference PCA model. This value is fed into a quadratic model fitted with the reference data set, which relates MSE values to levels of corruption artificially simulated by permuting entries from the reference set.<br>
#   * $d_{E, AE}$: computes the MSE obtained by reconstructing the new batch using the reference AE model. This value is fed into a quadratic model fitted with the reference data set, which relates MSE values to levels of corruption artificially simulated by permuting entries from the reference set.<br>
# 

# %% [markdown]
# ## Creation events<br>
# The following experiments use an initial subset as reference and add new batches of different size.

# %%
#print("  - Case 1: add rows of new set \n")
# FRO Added message to indicate more information about adding rows to the dataset
print("Step [4/6] of 6: Case 1 - Adding rows to the new data set =======")
print("Step [4/6]: Starting ...")
try:
    smexp_dyn.do_exp(X_NEW, ps_dict, resultspath = "../results/ds05", mode_der = "add_batch", 
                     batchsize_pctges = [0.05, 0.1, 0.25, 0.5, 0.75, 1], dstitle="DS 05 batch addition", resultsfile="/dyn-ds-05-add")
except Exception as e: print(e)

# FRO Added message to indicate that the adding rows to the dataset step has finished
print("Step [4/6]: Finished")

# %% [markdown]
# ## Update events<br>

# %%
#print("  - Case 2: transform columns ..\n")
# FRO Added message to indicate more information about adding rows to the dataset
print("Step [5/6] of 6: Case 2 - Modifying Columns in the data set =======")
print("Step [5/6]: Starting ...")
try: 
    smexp_dyn.do_exp(X_PS, ps_dict, resultspath = "../results/ds05", mode_der = "trans_cols",
                    tr_pctges = [0.05, 0.1, 0.3, 0.5, 0.7, 0.8, 1], dstitle="DS 05 cbrt scale", batchsize_pctges=[1], kfolds=1,
                    modetr="cbrt",resultsfile="/fixed-ds-05-trcols-cbrt")
except Exception as e: print(e)

# FRO Added message to indicate that the transforming columns rows to the dataset step has finished
print("SStep [5/6]: Finished")

# %% [markdown]
# ## Deletion of records<br>
# In the following cases, the reference set contains all the records and some of them are deleted in different ways: signals are down sampled, outliers are removed, etc.

# %%
#print("  - Case D: remove rows decimate .. \n")
# FRO Added message to indicate more information about adding rows to the dataset
print("Step [6/6] of 6: Case 3 - Removing Rows in the data set =======")
print("Step [6/6]: Starting ...")

try:
    smexp_dyn.do_exp(X_PS, ps_dict, resultspath = "../results/ds05", mode_der = "rem_rows_decimate", 
                            tr_pctges = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05], dstitle="DS 05 decimate", batchsize_pctges=[1], 
                            resultsfile="/fixed-ds-05-downsample")
except Exception as e: print(e)

# FRO Added message to indicate that the transforming columns rows to the dataset step has finished
print("Step [6/6]: Finished")
print("Please find the generated numeric and graphic results in the folder: /scientificDataPaper/results/ds05/")


