import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Wedge, Circle, Rectangle
from sympy.abc import alpha
from config import DATA_PATH

file_path = DATA_PATH
df = pd.read_excel(file_path)
df_numeric = df.select_dtypes(include=[np.number])

corr = df_numeric.corr(method='pearson')
labels = corr.columns.tolist()
n = len(labels)

plt.rcParams['font.family'] = 'Arial'
# Custom gradient: dark blue -> light blue -> white -> light red -> dark red
# colors_list = ["#8FB4BE", "#AFC9CF", "#D5E1E3", "#EBBFC2", "#E28187","#D93F49"]
colors_list = ["#24BFCA", "#7ECED6", "#D5E1E3", "#EBBFC2", "#E28187","#E77C8E"]
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors_list)

fig, ax = plt.subplots(figsize=(10, 10), dpi=500)
ax.set_aspect('equal')

grid_color = '#E5E5E5'

for i in range(n):
    for j in range(n):
        val = corr.iloc[i, j]
        norm_val = (val + 1) / 2
        color = cmap(norm_val)

        bg_color = color if i >= j else 'white'
        rect = Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=bg_color, edgecolor=grid_color, linewidth=0.8,alpha = 1)
        ax.add_patch(rect)

        if i >= j:
            ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                    color='black', fontsize=18, fontweight='bold', fontfamily='Arial')

        else:
            angle = abs(val) * 360
            wedge = Wedge((j, i), 0.35, 90 - angle / 2, 90 + angle / 2,
                          facecolor=color, edgecolor='none')
            ax.add_patch(wedge)
            ax.add_patch(Circle((j, i), 0.35, fill=False, edgecolor='#DDDDDD', linewidth=0.5))
ax.set_axisbelow(True)

ax.set_xlim(-0.5, n - 0.5)
ax.set_ylim(n - 0.5, -0.5)
ax.set_xticks(np.arange(n))
ax.set_yticks(np.arange(n))
ax.set_xticklabels(labels, fontsize=28, rotation=45, ha='right')
ax.set_yticklabels(labels, fontsize=28)

for spine in ax.spines.values():
    spine.set_visible(False)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=20)

plt.tight_layout()

ax.set_frame_on(True)
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)
    spine.set_color('black')

plt.savefig("pie_corr_matrix.png", bbox_inches='tight', dpi=900)
plt.show()