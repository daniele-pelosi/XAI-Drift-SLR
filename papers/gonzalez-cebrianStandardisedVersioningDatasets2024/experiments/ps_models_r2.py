"""
Script Name: ps_models_r2.py
Description: this module is for analysis and visualisation of the results related to 
Primary Source (PS) models in the context of the CUD experiments performed with each dataset. 
The script loads $R^2$ values and fitting time data for different datasets from Excel files
whose names must be provided manually. The script then generates a bar plot showcasing 
the $R^2$ values of (PS) models across multiple datasets, providing insights into model 
performance. Additionally, it creates a second bar plot illustrating the time required 
for fitting primary source models, categorized by different data drift metrics ($d_{P}$, 
$d_{E,PCA}$, $d_{E,AE}$). The resulting visualizations are saved as a PDF file for further 
analysis and dissemination.

Author:
- Name: Alba González-Cebrián, Fanny Rivera-Ortiz, Jorge Mario Cortés-Mendoza, Adriana E. Chis, Michael Bradford, Horacio González-Vélez
- Email: Alba.Gonzalez-Cebrian@ncirl.ie

License: MIT License
- License URL: https://opensource.org/license/mit/
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

resultspath = "../results/"

#%%
r2df = pd.concat([pd.read_excel(resultspath + "ds01/dyn-ds-01-PS-info.xlsx", sheet_name="r2", index_col=0),
                  pd.read_excel(resultspath + "ds02/dyn-ds-02-PS-info.xlsx", sheet_name="r2", index_col=0),
                  pd.read_excel(resultspath + "ds03/dyn-ds-03-PS-info.xlsx", sheet_name="r2", index_col=0),
                  pd.read_excel(resultspath + "ds04/dyn-ds-04-PS-info.xlsx", sheet_name="r2", index_col=0),
                  pd.read_excel(resultspath + "ds05/dyn-ds-05-PS-info.xlsx", sheet_name="r2", index_col=0),
                  pd.read_excel(resultspath + "ds06/dyn-ds-06-PS-info.xlsx", sheet_name="r2", index_col=0),
                  pd.read_excel(resultspath + "ds07/dyn-ds-07-PS-info.xlsx", sheet_name="r2", index_col=0)])
r2df.index = list(["DS 01", "DS 02", "DS 03", "DS 04", "DS 05", "DS 06", "DS 07"])
r2df = r2df.mask(r2df<0, 0)

timePSdf = pd.concat([pd.read_excel(resultspath + "ds01/dyn-ds-01-PS-info.xlsx", sheet_name="time", index_col=0),
                  pd.read_excel(resultspath + "ds02/dyn-ds-02-PS-info.xlsx", sheet_name="time", index_col=0),
                  pd.read_excel(resultspath + "ds03/dyn-ds-03-PS-info.xlsx", sheet_name="time", index_col=0),
                  pd.read_excel(resultspath + "ds04/dyn-ds-04-PS-info.xlsx", sheet_name="time", index_col=0),
                  pd.read_excel(resultspath + "ds05/dyn-ds-05-PS-info.xlsx", sheet_name="time", index_col=0),
                  pd.read_excel(resultspath + "ds06/dyn-ds-06-PS-info.xlsx", sheet_name="time", index_col=0),
                  pd.read_excel(resultspath + "ds07/dyn-ds-07-PS-info.xlsx", sheet_name="time", index_col=0)])
timePSdf.index = list(["DS 01", "DS 02", "DS 03", "DS 04", "DS 05", "DS 06", "DS 07"])
# 
#%%
fig, ax = plt.subplots(1, 2, figsize=(14,5))

r2df.plot.bar(rot=0, ax = ax[0], color=["k", "green"],  alpha=0.7)
ax[0].set_yticks(np.arange(0,1.1,0.1))
ax[0].grid(axis = "both")
ax[0].set_axisbelow(True)
ax[0].set_title("R$^2$ of primary source models")
ax[0].set_xlabel("Datasets")
ax[0].set_ylabel("$R^2$")
ax[0].legend(bbox_to_anchor =(0.5,-0.35), loc='lower center', ncol = 2, title = "Models")
ax[0].text(-0.1, 1.1, "a", transform = ax[0].transAxes, fontsize=14, weight = 'bold')

timePSdf.plot.bar(rot=0, ax = ax[1], color=["red", "blue", "green"], alpha=0.7)
ax[1].grid(axis = "both")
ax[1].set_yscale("log")
ax[1].set_axisbelow(True)
ax[1].set_title("Time for fitting primary source models")
ax[1].set_xlabel("Datasets")
ax[1].set_ylabel("s.")
ax[1].legend(labels=["$d_{P}$", "$d_{E,PCA}$", "$d_{E,AE}$"], 
             bbox_to_anchor =(0.5,-0.35), 
             loc='lower center', ncol = 3, title = "Data drift metrics")
ax[1].text(-0.1, 1.1, "b", transform = ax[1].transAxes, fontsize=14, weight = 'bold')
# %%
fig.tight_layout()
fig.savefig("../figures/PS_r2_time-new.pdf", dpi = 400, format="pdf")

# %%
