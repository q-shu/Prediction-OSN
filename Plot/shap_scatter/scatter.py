import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'

loaded_object = joblib.load('../../model/rej/model.joblib')
model_for_shap = loaded_object
df = pd.read_excel('../../data/data.xlsx')


y = df.iloc[:, -1]
# X = df.drop('Permeability (L m-2 h-1 bar-1) ', axis=1)
X = df.drop('Rejection (%)',axis=1)

explainer = shap.TreeExplainer(model_for_shap)
shap_values = explainer.shap_values(X)


styles =[
    ("#AFDEF3", "#2A3D66"),
    ("#A3E4D7", "#0E6655"),
    ("#D7BDE2", "#5B2C6F"),
    ("#ABEBC6", "#186A3B"),
    ("#F9E79F", "#7D6608"),
    ("#D6DBDF", "#2C3E50"),
    ("#FAD7A0", "#784212"),
    ("#AED6F1", "#1B4F72"),
    ("#A9DFBF", "#0B5345"),
]

for i in range(9):
    rank = i + 1
    mean_shap = np.abs(shap_values).mean(axis=0)

    sorted_indices = np.argsort(mean_shap)[::-1]

    target_index = sorted_indices[rank - 1]
    target_feature = X.columns[target_index]

    fill_color, edge_color = styles[i % len(styles)]

    print(f"Feature Importance Ranking No.{rank}: [{target_feature}]")

    plt.figure(figsize=(4.5, 3.5), dpi=600)

    shap.dependence_plot(
        target_feature,
        shap_values,
        X,
        interaction_index=None,
        color=fill_color,
        dot_size=20,
        alpha=1.0,
        show=False
    )

    ax = plt.gca()

    x_vals = X[target_feature].values
    y_vals = shap_values[:, X.columns.get_loc(target_feature)]
    ax.scatter(x_vals, y_vals,
               s=50,
               marker='o',
               c=fill_color,
               edgecolors=edge_color,
               linewidth=0.5,
               zorder=10)

    scatter = ax.collections[0]
    scatter.set_edgecolor('white')
    scatter.set_linewidth(0.5)

    ax.set_xlabel(target_feature, fontsize=28, fontfamily='Arial')
    ax.set_ylabel("SHAP Value", fontsize=28, fontfamily='Arial')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.2, alpha=0.5, zorder=0)

    ax.tick_params(axis='both', labelsize=24, direction='in', width=1.0, length=4)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.0)
        spine.set_color('black')

    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f'cat-{rank}-c.png', bbox_inches='tight', dpi=900)
    plt.show()