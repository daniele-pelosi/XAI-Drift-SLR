"""
Script Name: ds4_CUD_demo.py
Description: this script executes a demo of the Creation, Update and Deletion experiments 
with DS 04.

Author:
- Name: Alba González-Cebrián, Adriana E. Chis, Michael Bradford, Horacio González-Vélez
- Email: Alba.Gonzalez-Cebrian@ncirl.ie

License: MIT License
- License URL: https://opensource.org/license/mit/
"""
# %% [markdown]
# # Data set 4<br>
#  This dataset included a company's revenue, sales, and costs measured monthly from 2015 until March 2020. It is a fairly small and thin dataset compared to the other datasets, and it was included to test the performance of the ML models when the available records are scarce, which makes it a particularly interesting case study for deletion experiments.<br>
# ## Preliminaries<br>
# Import and load the uses Python packages and modules 

# %%
import sys
import os

# Add the parent directory to sys.path
parent_directory = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../main/'))
sys.path.insert(0, parent_directory)

import pandas as pd
import numpy as np
import sim_experiments as smexp_dyn
from datetime import date
import autoversion_service as av
import autoversion_aemod as avae

# %% [markdown]
# Load the data set and prepare it
print("Loading the Demo Configuration ...")

# %%
df = pd.read_csv("../datasets/ds04-Month_Value_1.csv", index_col="Period")
df.index = pd.to_datetime(df.index,format = '%d.%m.%Y')

# %%
X_PS = df.loc[df.index<"2019-01-01",:]
X_NEW = df.loc[df.index>="2019-01-01",:]

# %% [markdown]
# Exploratory analysis showing time series decomposigion as an additive model where each time instant ($x_i$) is the addition of a trend component ($T_i$), a seasonal component ($S_i$) and an error component ($E_i$)

# %%
smexp_dyn.plot_time_components_div(df, X_PS.index, X_NEW.index, 12, xplot_ps= X_PS.index,  xplot_rev= X_NEW.index, dsname="ds04comb", 
                                   xtxtsize=5, path_figs="../figures/")

# %% [markdown]
# ## Primary Source models<br>
import random
import tensorflow as tf
tf.random.set_seed(42) 
np.random.seed(42)
random.seed(42)
# Obtain the parameters for the reference batch of data. The function returns a dictionary *ps_dict* with the parameters to compute each one of the drift metrics according to a different ML model and the _indPS_ variable with the indices of the records used for the reference set.<br>
print("Doing Primary Source Model -- \n" + "N = " + str(len(X_PS)) + " (" + str(np.round(len(X_PS)/(len(X_PS) + len(X_NEW))*100,2)) + " %" + " of total dataset length)")
ps_dict, ind_PS = smexp_dyn.get_PS_data(X_PS, resultspath = "../results/ds04-demo/",  dstitle = "DS 04 PS", 
                                        PSfilename = "ds04ps.pkl", mse_model_type="splines", pctge_PS=1, resultsfile="dyn-ds-04")

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
# print("  - Case 1: add rows of new set \n")
# FRO Added message to indicate more information about adding rows to the dataset
print("Step [4/4]: Case 1 - Adding rows to the new data set =======")
print("Step [4/4]: Starting ...")

# Add batches of 10% New Dataset records' size
ds04_c_demo = smexp_dyn.do_exp(X_NEW, ps_dict, resultspath = "../results/ds04-demo", mod_pca = True, mod_ae = True, 
                               mode_der = "add_batch", batchsize_pctges = [0.2], demo = True,
                               dstitle="DS 04 creation-demo", kfolds=0, resultsfile="/demo-ds04-creation")

__, vC_pca_dp = av.versionDER(ds04_c_demo, ps_dict["PS_dic"]["PCA"]["S"], outdict=True, loadfile=False, dstitle="DS 04 creation-demo pca-dP", cat_threshold=2)
__, vC_pca_de = av.versionDER(ds04_c_demo, ps_dict["PS_dic"]["PCA"]["E"], outdict=True, loadfile=False, dstitle="DS 04 creation-demo pca-dE", drift_dist="dE", 
                                  cat_threshold = 2)
vC_ae_de = avae.versionDER(ds04_c_demo, ps_dict["PS_dic"]["AE"], "../results/ds04-demo/aelogs/tb_logs", "../results/ds04-demo", outdict=True, loadfile=False, 
                               dstitle="DS 04 creation-demo ae", cat_threshold = 2)

print("DS 04 - PS - v.tag: " + ps_dict["PS_dic"]["PCA"]["S"]["vtag"])
print("DS 04 - demo creation - PCA dP v.tag: " + vC_pca_dp["vtag"])
print("DS 04 - demo creation - PCA dE v.tag: " + vC_pca_de["vtag"])
print("DS 04 - demo creation - AE v.tag: " + vC_ae_de["vtag"])
# FRO Added message to indicate that the adding rows to the dataset step has finished
print("Step [4/4]: Finished.")
print("\n")

# %% [markdown]
# ## Update events<br>

# %%
#print("  - Case 2: transform columns ..\n")
# FRO Added message to indicate more information about adding rows to the dataset
print("Step [5/4]: Case 2 - Modifying Columns in the data set =======")
print("Step [5/4]: Starting ...")

ds04_u_demo = smexp_dyn.do_exp(X_PS, ps_dict, resultspath = "../results/ds04-demo",  mode_der = "trans_cols",
                               tr_pctges = [0.5], dstitle="DS 04 update-demo", batchsize_pctges=[1], kfolds=1,
                               modetr="cbrt",resultsfile="/demo-ds04-trcols-cbrt", demo = True)

__, vU_pca_dp = av.versionDER(ds04_u_demo, ps_dict["PS_dic"]["PCA"]["S"], outdict=True, loadfile=False, dstitle="DS 04 update-demo pca-dP", cat_threshold=2)
__, vU_pca_de = av.versionDER(ds04_u_demo, ps_dict["PS_dic"]["PCA"]["E"], outdict=True, loadfile=False, dstitle="DS 04 update-demo pca-dE", drift_dist="dE", 
                                  cat_threshold = 2)
vU_ae_de = avae.versionDER(ds04_u_demo, ps_dict["PS_dic"]["AE"], "../results/ds04-demo/aelogs/tb_logs", "../results/ds04-demo", outdict=True, loadfile=False,
                               dstitle="DS 04 update-demo ae", cat_threshold = 2)

print("DS 04 - PS - v.tag: " + ps_dict["PS_dic"]["PCA"]["S"]["vtag"])
print("DS 04 - demo update - PCA dP v.tag: " + vU_pca_dp["vtag"])
print("DS 04 - demo update - PCA dE v.tag: " + vU_pca_de["vtag"])
print("DS 04 - demo update - AE v.tag: " + vU_ae_de["vtag"])
# FRO Added message to indicate that the adding rows to the dataset step has finished
print("Step [5/4]: Finished.")
print("\n")

# %% [markdown]
# ## Deletion events<br>
# In the following cases, the reference set contains all the records and some of them are deleted in different ways: signals are down sampled, outliers are removed, etc.

# %%
#print("  - Case 3: remove rows decimate .. \n")
# FRO Added message to indicate more information about adding rows to the dataset
print("Step [6/4]: Case 3 - Removing Rows in the data set =======")
print("Step [6/4]: Starting ...")

ds04_d_demo = smexp_dyn.do_exp(X_PS, ps_dict, resultspath = "../results/ds04-demo",  mode_der = "rem_rows_decimate", tr_pctges = [0.5], dstitle="DS 04 deletion-demo", 
                               batchsize_pctges=[1], resultsfile="/demo-ds04-deletion", demo = True)

__, vD_pca_dp = av.versionDER(ds04_d_demo, ps_dict["PS_dic"]["PCA"]["S"], outdict=True, loadfile=False, dstitle="DS 04 deletion-demo pca-dP", cat_threshold=2)
__, vD_pca_de = av.versionDER(ds04_d_demo, ps_dict["PS_dic"]["PCA"]["E"], outdict=True, loadfile=False, dstitle="DS 04 deletion-demo pca-dE", drift_dist="dE", 
                                  cat_threshold = 2)
vD_ae_de = avae.versionDER(ds04_d_demo, ps_dict["PS_dic"]["AE"], "../results/ds04-demo/aelogs/tb_logs", "../results/ds04-demo", outdict=True, loadfile=False, 
                               dstitle="DS 04 deletion-demo ae", cat_threshold = 2)

print("DS 04 - PS - v.tag: " + ps_dict["PS_dic"]["PCA"]["S"]["vtag"])
print("DS 04 - demo deletion - PCA dP v.tag: " + vD_pca_dp["vtag"])
print("DS 04 - demo deletion - PCA dE v.tag: " + vD_pca_de["vtag"])
print("DS 04 - demo deletion - AE v.tag: " + vD_ae_de["vtag"])

# %%
demo_results = pd.DataFrame({"datasets": "ds04-demo", 
                "PS": {"dPCA_P": ps_dict["PS_dic"]["PCA"]["S"]["vtag"], "dPCA_E":  ps_dict["PS_dic"]["PCA"]["E"]["vtag"], 
                       "dAE_E": ps_dict["PS_dic"]["AE"]["vtag"]},
                "creation": {"dPCA_P": vC_pca_dp["vtag"], "dPCA_E": vC_pca_de["vtag"], "dAE_E": vC_ae_de["vtag"]},
                "update": {"dPCA_P": vU_pca_dp["vtag"], "dPCA_E": vU_pca_de["vtag"], "dAE_E": vU_ae_de["vtag"]}, 
                "deletion": {"dPCA_P": vD_pca_dp["vtag"], "dPCA_E": vD_pca_de["vtag"], "dAE_E": vD_ae_de["vtag"]}})
if os.path.exists("../results/demos.xlsx"):
       with pd.ExcelWriter("../results/demos.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
              demo_results.to_excel(writer, sheet_name="ds04")
else:
       demo_results.to_excel("../results/demos.xlsx", sheet_name='ds04')  

# FRO Added message to indicate that the adding rows to the dataset step has finished
print("Step [6/4]: Finished.")
print("Please find the generated numeric and graphic results in the folder: /scientificDataPaper/results/ds04-demo/")
