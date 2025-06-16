"""
Script Name: plot_paper_figures-add-batches.py
Description: this script is designed for the analysis and visualization of results related 
to the creation events in the context of CUD experiments performed with each dataset. 
The script loads $R^2$ values and fitting time data from Excel files, requiring manual 
input of file paths. It then generates a series of subplots, each depicting the data drift 
metrics ($d_{P}$, $d_{E, PCA}$, $d_{E, AE}$) over batch iterations for different datasets. 
The script utilizes custom line styles and colors to distinguish between the metrics and datasets, 
which are also provided manually and can be set by the user. The resulting visualizations 
are saved as a PDF file for further analysis and dissemination. 

Author:
- Name: Alba González-Cebrián, Fanny Rivera-Ortiz, Jorge Mario Cortés-Mendoza, Adriana E. Chis, Michael Bradford, Horacio González-Vélez
- Email: Alba.Gonzalez-Cebrian@ncirl.ie

License: MIT License
- License URL: https://opensource.org/license/mit/
"""
# %%
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
# %%
figsize_drift_v = (5, 2.5)
figsize_reps_v = (5, 2.5)
bbox_to_anchor_v = (2, 0.8, 0.1, 0.05)

# Create the figure and subplots
fig, axs = plt.subplots(2, 4, figsize=(15, 6), )

# File paths and labels
file_paths = [
    "../results/ds01/dyn-ds-01-add.xlsx",
    "../results/ds02/dyn-ds-02-add.xlsx",
    "../results/ds03/dyn-ds-03-add.xlsx",
    "../results/ds04/dyn-ds-04-add.xlsx",
    "../results/ds05/dyn-ds-05-add.xlsx",
    "../results/ds06/dyn-ds-06-add.xlsx",
    "../results/ds07/dyn-ds-07-add.xlsx"
]

ds_names = ["01", "02", "03", "04", "05", "06", "07"]

custom_lines_drifts_seq = [Line2D([0], [0], color="red", lw=1.5, linestyle="-"), 
                        Line2D([0], [0], color="blue", lw=1.5, linestyle="--"),
                        Line2D([0], [0], color="green", lw=1.5, linestyle=":")]

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
n_series = [1, 1, 1, 2, 1, 1, 1]
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
    
    # Plot repetitions
    ax.plot(performance_series["dP"].iloc[n_series[i],:], color="red", linestyle="-", alpha =1)
    ax.plot(performance_series["dEPCA"].iloc[n_series[i],:], color="blue", linestyle="--", alpha = 1)
    ax.plot(performance_series["dEAE"].iloc[n_series[i],:], color="green", linestyle=":", alpha = 1)
    ax.grid()
    ax.set_ylim(-5,105)
    ax.set_xlabel("Batch iterations")
    ax.set_title("DS " + ds_names[i] + " batches (" + str(int(col_names[n_series[i]] *100))+  "% mem. size)")
    ax.set_ylabel("data drift")

    # Label each panel with a lowercase letter
    ax.text(-0.2, 1.1, label, transform = ax.transAxes, fontsize=14, weight = 'bold')

fig.delaxes(axs.flat[7])

# Show the figure
fig.suptitle("Creation events (results by batch)", fontsize=16)

# Adjust spacing between subplots
fig.tight_layout()

# Adjust spacing between subplots
lgd = ax.legend(custom_lines_drifts_seq, ['$d_{P}$', '$d_{E, PCA}$', '$d_{E, AE}$'], bbox_to_anchor = bbox_to_anchor_v, 
                     loc="right", ncols=1, title = "Data drift metrics")
plt.show(block=False)
# FRO: Added the following line to remove the need to press a key
#plt.pause(0.1)
fig.savefig("../figures/creation-datadrift-batches.pdf", dpi = 400, format="pdf")
# %%
