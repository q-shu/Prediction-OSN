import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.inspection import partial_dependence
import joblib

plt.rcParams['font.family'] = 'Arial'
# ==========================================
# 1. Importing Data and Models
# ==========================================
df = pd.read_excel('../../data/data.xlsx')
# df = pd.read_excel('../../data/permeability-use-4.xlsx')

columns_to_keep = ['Contact angle (°)','Pore diameter (nm)','MWCO (Da)','RMS roughness (nm)','Film thickness (nm)', 'Pressure (bar)', 'Temperature (°C)','Concentration (mg/L)','Solute MW (g/mol) ','Rejection (%)']
# columns_to_keep = ['Contact angle (°)','Pore diameter dp (nm)','MWCO (Da)','RMS roughness (nm)','Film thickness (nm)', 'Pressure (bar)', 'Temperature (°C)','Concentration (mg/L)','Solute MW (g/mol) ','Permeability (L m-2 h-1 bar-1) ']

X_data = df[columns_to_keep]

model = joblib.load('../../model/rej/model.joblib')
# model = joblib.load('../../model/per/model.joblib')

feature_y = 'MWCO (Da)'
feature_x = 'Pore diameter (nm)'
# feature_y = 'Solute MW (g/mol) '

# ==========================================
# 2. Compute the 2D PDP matrix
# ==========================================

pd_results = partial_dependence(
    estimator=model,
    X=X_data,
    features=[feature_x, feature_y],
    grid_resolution=80
)

if 'grid_values' in pd_results:
    x_grid = pd_results['grid_values'][0]
    y_grid = pd_results['grid_values'][1]
else:
    x_grid = pd_results['values'][0]
    y_grid = pd_results['values'][1]
X, Y = np.meshgrid(x_grid, y_grid)
Z = pd_results['average'][0].T

# ==========================================
# 3. Drawing Section
# ==========================================
z_min, z_max = np.min(Z), np.max(Z)

min_idx = np.unravel_index(np.argmin(Z), Z.shape)
max_idx = np.unravel_index(np.argmax(Z), Z.shape)
x_min, y_min = X[min_idx], Y[min_idx]
x_max, y_max = X[max_idx], Y[max_idx]

fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
cf = ax.contourf(X, Y, Z, levels=100, cmap='RdYlBu_r', alpha=0.9)

ax.tick_params(axis='both', labelsize=24)

# Add Color Labels and a Legend
cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=24)

ax.set_xlabel(feature_x, fontsize=32)
ax.set_ylabel(feature_y, fontsize=32)

plt.tight_layout()
plt.savefig('M-P.png',dpi = 900)

plt.show()