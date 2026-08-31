import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Preparations
# ==========================================
plt.rcParams['font.family'] = 'Arial'

loaded_object = joblib.load('../../model/rej/R.joblib')

model_for_shap = loaded_object


df = pd.read_excel('../../data/data.xlsx')


y = df.iloc[:, -1]
X = df.drop('Rejection (%)', axis=1)
# X = df.drop('Permeability (L m-2 h-1 bar-1) ',axis=1)

# ==========================================
# 2. Calculate SHAP Values
# ==========================================
explainer = shap.TreeExplainer(model_for_shap)
shap_values = explainer.shap_values(X)


# ==========================================
# 3. Visual Analysis
# ==========================================
plt.figure(figsize=(10, 6),dpi=600)

shap.summary_plot(shap_values, X, show=False)

fig = plt.gcf()
ax = plt.gca()

ax.tick_params(axis='y', labelsize=20)

ax.tick_params(axis='x', labelsize=18)

ax.set_xlabel(ax.get_xlabel(), fontsize=18, fontfamily='Arial')

# 4. Adjust the Colorbar font size
if len(fig.axes) > 1:
    cbar_ax = fig.axes[-1]

    cbar_ax.tick_params(labelsize=18)

    cbar_ax.set_ylabel(cbar_ax.get_ylabel(), fontsize=20, fontfamily='Arial',labelpad=-25)

plt.tight_layout()
plt.savefig('../../pic/beeswarm.png', bbox_inches='tight', dpi = 900)
plt.show()