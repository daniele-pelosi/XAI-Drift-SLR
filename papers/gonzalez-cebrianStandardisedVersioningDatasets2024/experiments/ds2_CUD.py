"""
Script Name: ds2_CUD.py
Description: this script executes the Creation, Update and Deletion experiments with DS 02.

Author:
- Name: Alba González-Cebrián, Adriana E. Chis, Michael Bradford, Horacio González-Vélez
- Email: Alba.Gonzalez-Cebrian@ncirl.ie

License: MIT License
- License URL: https://opensource.org/license/mit/
"""
# %% [markdown]
# # Data set 2<br>
# This dataset consists of time series of reported cases of chickenpox at the county level between 2005 and 2015. The dataset can be used for both county-level and nation-level case count prediction. For this work, the counts per county were used. For the experiments involving the addition of rows, the Primary Source contained records from 2005 to 2013, and the last two years were treated as a new block of records.<br>
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
df = pd.read_csv("../datasets/ds02-hungary_chickenpox.csv")
df["Date"] = pd.to_datetime(df.Date, format = '%d/%m/%Y')
dfnum = df.select_dtypes(include=[float, int])
dfnum.index = df["Date"]
n_samples_ps = np.min(np.where(df["Date"]>="2010"))
# %%
X_PS = dfnum.iloc[:n_samples_ps+1,]
X_NEW = dfnum.iloc[n_samples_ps+1:,]
# %% [markdown]
# Exploratory analysis showing time series decomposigion as an additive model where each time instant ($x_i$) is the addition of a trend component ($T_i$), a seasonal component ($S_i$) and an error component ($E_i$)

# %%
smexp_dyn.plot_time_components_div(dfnum, X_PS.index, X_NEW.index, 52, xplot_ps=X_PS.index, xplot_rev=X_NEW.index, dsname="ds02comb", xtxtsize=5, path_figs="../figures/")

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
ps_dict, ind_PS = smexp_dyn.get_PS_data(X_PS, resultspath = "../results/ds02/", dstitle = "DS 02 PS", 
                                        PSfilename = "ds02ps.pkl", mse_model_type="splines", pctge_PS=1, resultsfile="dyn-ds-02")

# %% [markdown]
# ### New versions<br>
# When a new version of the dataset is generated, it will be compared to the information from the previous version in the following way:<br>
#   * $d_{P}$: computes the cosine distance between loading matrices obtained for both data sets;<br>
#   * $d_{E, PCA}$: computes the MSE obtained by reconstructing the new batch using the reference PCA model. This value is fed into a quadratic model fitted with the reference data set, which relates MSE values to levels of corruption artificially simulated by permuting entries from the reference set.<br>
#   * $d_{E, AE}$: computes the MSE obtained by reconstructing the new batch using the reference AE model. This value is fed into a quadratic model fitted with the reference data set, which relates MSE values to levels of corruption artificially simulated by permuting entries from the reference set.<br>
# 

# %% [markdown]
# ## Creation events<br>
# The following experiments use an initial subset as reference, emulating the scenario of dynamic update with batches of different size.

# %%
#print("  - Case 1: add rows of new set \n")
# FRO Added message to indicate more information about adding rows to the dataset
print("Step [4/6]: Case 1 - Adding rows to the new data set =======")
print("Step [4/6]: Starting ...")
try: 
    smexp_dyn.do_exp(X_NEW, ps_dict, resultspath = "../results/ds02", mode_der = "add_batch", 
                    batchsize_pctges = [0.05, 0.1, 0.25, 0.5, 0.75, 1], dstitle="DS 02 batch addition", resultsfile="/dyn-ds-02-add")
except Exception as e: print(e)

# FRO Added message to indicate that the adding rows to the dataset step has finished
print("Step [4/6]: Finished")

# %% [markdown]
# ## Update events<br>

# %%
#print("  - Case 2: transform columns ..\n")
# FRO Added message to indicate more information about adding rows to the dataset
print("Step [5/6]: Case 2 - Modifying Columns in the data set =======")
print("Step [5/6]: Starting ...")
try: 
    smexp_dyn.do_exp(X_PS, ps_dict, resultspath = "../results/ds02", mode_der = "trans_cols",
                    tr_pctges = [0.05, 0.1, 0.3, 0.5, 0.7, 0.8, 1], dstitle="DS 02 cbrt scale", batchsize_pctges=[1], kfolds=1,
                    modetr="cbrt",resultsfile="/fixed-ds-02-trcols-cbrt")
except Exception as e: print(e)

# FRO Added message to indicate that the transforming columns rows to the dataset step has finished
print("Step [5/6]: Finished")

# %% [markdown]
# ## Deletion events<br>
# In the following cases, the reference set contains all the records and some of them are deleted in different ways: signals are down sampled, outliers are removed, etc.

# %%
#print("  - Case 3: remove rows decimate .. \n")
# FRO Added message to indicate more information about adding rows to the dataset
print("Step [6/6]: Case 3 - Removing Rows in the data set =======")
print("Step [6/6]: Starting ...")
try:
    smexp_dyn.do_exp(X_PS, ps_dict, resultspath = "../results/ds02", mode_der = "rem_rows_decimate", 
                    tr_pctges = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05], dstitle="DS 02 decimate", batchsize_pctges=[1], 
                    resultsfile="/fixed-ds-02-downsample")
except Exception as e: print(e)

# FRO Added message to indicate that the transforming columns rows to the dataset step has finished
print("Step [6/6]: Finished")
print("Please find the generated numeric and graphic results in the folder: /scientificDataPaper/results/ds02/")
