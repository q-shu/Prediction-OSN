import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Set plot style
sns.set_theme(style="white")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 20


# 2. Prepare data
# Loading dataset from the specified path
# df = pd.read_excel('../data/your_data.xlsx')
df = pd.read_excel('../data/your_data.xlsx')

# Select only numeric columns for correlation calculation
df_numeric = df.select_dtypes(include=[np.number])

# 3. Calculate Pearson correlation matrix
corr = df_numeric.corr(method='pearson')

# 4. Generate a mask to hide the upper triangle of the heatmap
mask = np.triu(np.ones_like(corr, dtype=bool))

# 5. Initialize figure size
f, ax = plt.subplots(figsize=(11, 9))

# 6. Define the color palette
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# 7. Draw the heatmap
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
            )          # Hide numeric annotations as requested

# 8. Set title and axis labels
# plt.title('Pearson Correlation Matrix Of Rejection', fontsize=20, pad=20)
# plt.title('Pearson Correlation Matrix Of Permeability', fontsize=20, pad=20)
plt.xticks(rotation=45, ha='right',fontsize=14)
plt.yticks(rotation=0,fontsize=18)
ax.tick_params(left=False, bottom=False)
# 9. Save and Show
plt.tight_layout()

# Ensure the save directory exists
save_dir = '../pic/pearson'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# plt.savefig(os.path.join(save_dir, 'pearson_rej.svg'))
plt.savefig(os.path.join(save_dir, 'pearson_per.svg'))
plt.show()