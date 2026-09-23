import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from config import DATA_PATH

sns.set_theme(style="white")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 20

df = pd.read_excel(DATA_PATH)
df_numeric = df.select_dtypes(include=[np.number])
corr = df_numeric.corr(method='pearson')
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))


cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(corr,
            mask=mask,            # Apply the upper triangle mask
            cmap=cmap,            # Apply the color palette
            vmax=1, vmin=-1,      # Lock color range from -1 to 1
            center=0,             # Set center point to 0 (white)
            square=True,          # Force cells to be square
            linewidths=.5,        # Width of the lines between cells
            cbar_kws={"shrink": .7}, # Shrink colorbar for better aesthetics
            # annot=False,
            annot=False,
            fmt=".2f",
            annot_kws={
                "size": 18,
                "family": "Arial",
                # "weight": "bold"
            },
            # xticklabels=False,
            # yticklabels=False
            )

# plt.title('Pearson Correlation Matrix Of Rejection', fontsize=20, pad=20)
# plt.title('Pearson Correlation Matrix Of Permeability', fontsize=20, pad=20)
plt.xticks(rotation=45, ha='right',fontsize=14)
plt.yticks(rotation=0,fontsize=18)
ax.tick_params(left=False, bottom=False)
plt.tight_layout()

save_dir = '../pic/pearson'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

plt.savefig(os.path.join(save_dir, 'pearson_per.svg'))
plt.show()