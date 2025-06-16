"""
Script Name: ds3_CUD.py
Description: this script executes the Creation, Update and Deletion experiments with DS 03.

Author:
- Name: Alba González-Cebrián, Adriana E. Chis, Michael Bradford, Horacio González-Vélez
- Email: Alba.Gonzalez-Cebrian@ncirl.ie

License: MIT License
- License URL: https://opensource.org/license/mit/
"""
# %% [markdown]
# # Data set 3<br>
# The original source of this dataset contained monthly measurements of global land temperatures by country, reported between 1743 and 2013. The measurements were aggregated by the average and variability of temperature measurements within each country each month. Due to a high rate of missing values in previous years, the data used for our experiments started in 1900. The Primary Source dataset included data from 1900 to 1999, and the measurements from this century constituted the block of new records. <br>
# ## Preliminaries

# %% [markdown]
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
data= pd.read_csv("../datasets/ds03-GlobalLandTemperaturesByCountry.csv")
df = data.pivot(columns="Country", values=["AverageTemperature", "AverageTemperatureUncertainty"], index="dt")
df_1900 = df.loc[(df.index>="1900-01-01") & (df.index<="1999-31-12"),:]
df_2000 = df.loc[df.index > "1999-31-12",:]

# %% [markdown]
# Exploratory analysis showing time series decomposigion as an additive model where each time instant ($x_i$) is the addition of a trend component ($T_i$), a seasonal component ($S_i$) and an error component ($E_i$). There are high percentages of missing values in this dataset, so linear interpolation is performed prior to the exploratory analysis of the time series components

# %%
redmid_ds = pd.concat([df_1900.iloc[-500:,:], df_2000]).interpolate(method="linear")
red_ds = pd.concat([df_1900, df_2000]).interpolate(method="linear")
smexp_dyn.plot_time_components_div(red_ds, df_1900.index, df_2000.index, 12, xplot_ps=df_1900.index, xplot_rev=df_2000.index, dsname="ds03comb",
                                   xtxtsize=5, path_figs="../figures/")

# %% [markdown]
# ## Primary Source models<br>
# Obtain the parameters for the reference batch of data. The function returns a dictionary *ps_dict* with the parameters to compute each one of the drift metrics according to a different ML model and the _indPS_ variable with the indices of the records used for the reference set.

# %%

import random
import tensorflow as tf
tf.random.set_seed(42) 
np.random.seed(42)
random.seed(42)
print("Doing Primary Source Model -- \n" + "N = " + str(len(df_1900)) + " (" + str(np.round(len(df_1900)/(len(df_1900) + len(df_2000))*100,2)) + " %" + " of total dataset length)")
ps_dict, ind_PS = smexp_dyn.get_PS_data(df_1900, resultspath = "../results/ds03/", dstitle = "DS 03 PS", PSfilename = "ds03ps.pkl", pctge_PS=1,
                                        mse_model_type="splines", tr_val_split=0.69, resultsfile =  "/dyn-ds-03")

# %%
X_PS = df.loc[list(ind_PS),:].copy()
X_NEW = df_2000

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
print("Step [4/6]: Case 1 - Adding rows to the new data set =======")
print("Step [4/6]: Starting ...")
try:
    smexp_dyn.do_exp(X_NEW, ps_dict, resultspath = "../results/ds03", mode_der = "add_batch", batchsize_pctges = [0.05, 0.1, 0.25, 0.5, 0.75, 1],
                    dstitle="DS 03 batch addition", resultsfile="/dyn-ds-03-add")
except Exception as e: print(e)
# FRO Added message to indicate that the adding rows to the dataset step has finished
print("Step [4/6]: Finished")

# %% [markdown]
# ## Update events<br>

# %%
#print("  - Case 2.1: transform columns ..\n")
# FRO Added message to indicate more information about adding rows to the dataset
print("Step [5/6]: Case 2 - Modifying Columns in the data set =======")
print("Step [5/6]: Starting ...")
try: 
    smexp_dyn.do_exp(X_PS, ps_dict, resultspath = "../results/ds03", mode_der = "trans_cols",
                    tr_pctges = [0.05, 0.1, 0.3, 0.5, 0.7, 0.8, 1], dstitle="DS 03 cbrt scale", batchsize_pctges=[1], kfolds=1,
                    modetr="cbrt",resultsfile="/fixed-ds-03-trcols-cbrt")
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
    smexp_dyn.do_exp(X_PS, ps_dict, resultspath = "../results/ds03", mode_der = "rem_rows_decimate",
                            tr_pctges= [0.5, 0.4, 0.3, 0.2, 0.1, 0.05], dstitle="DS 03 decimate", resultsfile="/fixed-ds-03-downsample",
                            batchsize_pctges=[1])
except Exception as e: print(e)

# FRO Added message to indicate that the transforming columns rows to the dataset step has finished
print("Step [6/6]: Finished")
print("Please find the generated numeric and graphic results in the folder: /scientificDataPaper/results/ds03/")


