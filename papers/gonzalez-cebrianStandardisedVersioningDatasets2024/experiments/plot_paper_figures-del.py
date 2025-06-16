"""
Script Name: plot_paper_figures-del.py
Description: this script is designed for the analysis and visualization of results related 
to the deletion events in the context of CUD experiments performed with each dataset. 
The script loads $R^2$ values and fitting time data from Excel files, requiring manual 
input of file paths. It then generates two sets of subplots for different metrics: 
the first set illustrates data drift metrics ($d_{P}$, $d_{E, PCA}$, $d_{E, AE}$) over batch 
iterations, summarising the values by the mean, 5% and 95% percentiles; while the second 
set depicts the time required for the fitting process. The script utilizes custom line styles 
and colors to distinguish between the metrics and datasets, which are also provided manually 
and can be set by the user. The resulting visualizations are saved as a PDF file for further 
analysis and dissemination. 

Author:
- Name: Alba González-Cebrián, Fanny Rivera-Ortiz, Jorge Mario Cortés-Mendoza, Adriana E. Chis, Michael Bradford, Horacio González-Vélez
- Email: Alba.Gonzalez-Cebrian@ncirl.ie

License: MIT License
- License URL: https://opensource.org/license/mit/
"""
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
# %%
figsize_drift_v = (4, 2.5)
figsize_reps_v = (4, 2.5)
bbox_to_anchor_v = (2, 0.8, 0.1, 0.05)

# File paths and labels
file_paths = [
    "../results/ds01/fixed-ds-01-downsample.xlsx",
    "../results/ds02/fixed-ds-02-downsample.xlsx",
    "../results/ds03/fixed-ds-03-downsample.xlsx",
    "../results/ds04/fixed-ds-04-downsample.xlsx",
    "../results/ds05/fixed-ds-05-downsample.xlsx",
    "../results/ds06/fixed-ds-06-downsample.xlsx",
    "../results/ds07/fixed-ds-07-downsample.xlsx"
]
ds_names = ["01", "02", "03", "04", "05", "06", "07"]

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
# %% 
# Create the figure and subplots
fig, axs = plt.subplots(2, 4, figsize=(12.5, 6))
# Generate data for each subplot
for i, (ax, file_path, label) in enumerate(zip(axs.flat, file_paths, labels)):
    # Replace the following lines with your own data loading and plotting code
    performance_series = {"dP": pd.read_excel(file_path, sheet_name='drift-dP', index_col=0), 
                          "dEPCA": pd.read_excel(file_path, sheet_name='drift-dE_PCA', index_col=0), 
                          "dEAE": pd.read_excel(file_path, sheet_name='drift-dE_AE', index_col=0),
                          "tP": pd.read_excel(file_path, sheet_name= 'time-dP', index_col=0), 
                          "tEPCA": pd.read_excel(file_path, sheet_name='time-dE_PCA', index_col=0), 
                          "tEAE": pd.read_excel(file_path, sheet_name='time-dE_AE', index_col=0)}
    col_names = list(performance_series["dP"].index.values)
    col_names = [float(x.replace("level ", "") )for x in col_names]
    plt.rc("axes", titlesize = 10, labelsize = 9)
    
    # Plot data drift metrics
    custom_lines_drifts = [Line2D([0], [0], color="red", lw=1.5, linestyle="-", marker="o", markeredgecolor="black", markerfacecolor="white"), 
                           Line2D([0], [0], color="blue", lw=1.5, linestyle="--", marker="o", markeredgecolor="black", markerfacecolor="white"),
                           Line2D([0], [0], color="green", lw=1.5, linestyle=":", marker="s", markeredgecolor="black", markerfacecolor="white")]
    
    
    ax.fill_between(col_names, np.nanpercentile(performance_series["dP"], 5, axis=1), np.nanpercentile(performance_series["dP"], 95, axis=1), 
                      alpha = 0.2, color = "r", label='_nolegend_')
    ax.fill_between(col_names, np.nanpercentile(performance_series["dEPCA"], 5, axis=1), np.nanpercentile(performance_series["dEPCA"], 95, axis=1), 
                      alpha = 0.2, color = "b", label='_nolegend_')
    ax.fill_between(col_names, np.nanpercentile(performance_series["dEAE"], 5, axis=1), np.nanpercentile(performance_series["dEAE"], 95, axis=1), 
                      alpha = 0.2, color = "green", label='_nolegend_')
    
    ax.plot(col_names, performance_series["dP"].mean(axis=1), color="red", linestyle="-", marker="o", linewidth=2, alpha = 1, 
              markeredgecolor="black", markerfacecolor="white")
    ax.plot(col_names, performance_series["dEPCA"].mean(axis=1), color="blue", linestyle="--", marker="o", linewidth=2, alpha = 1, 
              markeredgecolor="black", markerfacecolor="white")
    ax.plot(col_names, performance_series["dEAE"].mean(axis=1), color="green", linestyle=":",  marker="s", linewidth=2, alpha = 1, 
              markeredgecolor="black", markerfacecolor="white")
    ax.grid()
    
    ax.set_xticks(col_names, [str(int(x*100))for x in col_names])
    ax.set_ylim(-5,105)
    ax.set_xlabel("Mem. size (%)")
    ax.set_title("DS " + ds_names[i])
    ax.set_ylabel("data drift")
    
    # Label each panel with a lowercase letter
    ax.text(-0.2, 1.1, label, transform = ax.transAxes, fontsize=14, weight = 'bold')

fig.delaxes(axs.flat[7])
fig.suptitle("Deletion events", fontsize=16)

# Adjust spacing between subplots
fig.tight_layout()
lgd = ax.legend(custom_lines_drifts, ['$d_{P}$', '$d_{E, PCA}$', '$d_{E, AE}$'], bbox_to_anchor = bbox_to_anchor_v, 
                     loc="right", ncols=1, title = "Data drift metrics")

# Show the figure
plt.show()
# FRO: Added the following line to remove the need to press a key
plt.pause(0.1)
fig.savefig("../figures/deletions-datadrift.pdf", dpi = 400, format="pdf")
# %%
# Create the figure and subplots
fig, axs = plt.subplots(2, 4, figsize=(12.5, 6))
# Generate data for each subplot
for i, (ax, file_path, label) in enumerate(zip(axs.flat, file_paths, labels)):
    # Replace the following lines with your own data loading and plotting code
    performance_series = {"dP": pd.read_excel(file_path, sheet_name='drift-dP', index_col=0), 
                          "dEPCA": pd.read_excel(file_path, sheet_name='drift-dE_PCA', index_col=0), 
                          "dEAE": pd.read_excel(file_path, sheet_name='drift-dE_AE', index_col=0),
                          "tP": pd.read_excel(file_path, sheet_name= 'time-dP', index_col=0), 
                          "tEPCA": pd.read_excel(file_path, sheet_name='time-dE_PCA', index_col=0), 
                          "tEAE": pd.read_excel(file_path, sheet_name='time-dE_AE', index_col=0)}
    col_names = list(performance_series["dP"].index.values)
    col_names = [float(x.replace("level ", "") )for x in col_names]
    plt.rc("axes", titlesize = 10, labelsize = 9)
    
    # Plot data drift metrics
    custom_lines_drifts = [Line2D([0], [0], color="red", lw=1.5, linestyle="-", marker="o", markeredgecolor="black", markerfacecolor="white"), 
                           Line2D([0], [0], color="blue", lw=1.5, linestyle="--", marker="o", markeredgecolor="black", markerfacecolor="white"),
                           Line2D([0], [0], color="green", lw=1.5, linestyle=":", marker="s", markeredgecolor="black", markerfacecolor="white")]
    
    
    ax.fill_between(col_names, np.nanpercentile(performance_series["tP"], 5, axis=1), np.nanpercentile(performance_series["tP"], 95, axis=1), 
                      alpha = 0.2, color = "r", label='_nolegend_')
    ax.fill_between(col_names, np.nanpercentile(performance_series["tEPCA"], 5, axis=1), np.nanpercentile(performance_series["tEPCA"], 95, axis=1), 
                      alpha = 0.2, color = "b", label='_nolegend_')
    ax.fill_between(col_names, np.nanpercentile(performance_series["tEAE"], 5, axis=1), np.nanpercentile(performance_series["tEAE"], 95, axis=1), 
                      alpha = 0.2, color = "green", label='_nolegend_')
    
    ax.plot(col_names, performance_series["tP"].mean(axis=1), color="red", linestyle="-", marker="o", linewidth=2, alpha = 1, 
              markeredgecolor="black", markerfacecolor="white")
    ax.plot(col_names, performance_series["tEPCA"].mean(axis=1), color="blue", linestyle="--", marker="o", linewidth=2, alpha = 1, 
              markeredgecolor="black", markerfacecolor="white")
    ax.plot(col_names, performance_series["tEAE"].mean(axis=1), color="green", linestyle=":",  marker="s", linewidth=2, alpha = 1, 
              markeredgecolor="black", markerfacecolor="white")
    ax.grid()
    
    ax.set_xticks(col_names, [str(int(x*100))for x in col_names])
    ax.set_xlabel("Mem. size (%)")
    ax.set_title("DS " + ds_names[i])
    ax.set_ylabel("time (s.)")
    # Label each panel with a lowercase letter
    ax.text(-0.2, 1.1, label, transform = ax.transAxes, fontsize=14, weight = 'bold')

fig.delaxes(axs.flat[7])
        
fig.suptitle("Deletion events", fontsize=16)
# Adjust spacing between subplots
fig.tight_layout()
lgd = ax.legend(custom_lines_drifts, ['$d_{P}$', '$d_{E, PCA}$', '$d_{E, AE}$'], bbox_to_anchor = bbox_to_anchor_v, 
                loc="right", ncols=1, title = "Data drift metrics")

# Show the figure
plt.show(block=False)
# FRO: Added the following line to remove the need to press a key
#plt.pause(0.1)
fig.savefig("../figures/deletion-time.pdf", dpi = 400, format="pdf")

# %%
