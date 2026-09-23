import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

groups = ['$R^2$', 'MAE', 'RMSE']
features = ['CatBoost', 'LightGBM', 'RF', 'XGBoost', 'SVM']

# raw_values_rej = [[0.80, 0.71, 0.54, 0.64, 0.54], [2.65, 3.67, 4.61, 3.90, 4.62], [5.71, 6.92, 8.65, 7.67, 8.62]]
raw_values_per = [[0.82, 0.77, 0.79, 0.77, 0.66], [3.07, 3.58, 3.30, 3.47, 3.68], [4.21, 4.88, 4.67, 4.91, 5.87]]

# colors = ['#E3AE98', '#D9B1B7', '#ABD8C9']  Orange, Red, Green
colors = ['#D9B1B7', '#ABD8C9', '#E3AE98']

plot_values = []
for group in raw_values_per:
    group_max = max(group)
    normalized_group = [v / group_max for v in group]
    plot_values.append(normalized_group)

plt.rcParams['figure.dpi'] = 600
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Arial']

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'polar': True})
ax.set_theta_direction(-1)
ax.set_theta_offset(np.pi / 2)

ax.set_ylim(0, 1.25)
ax.set_rorigin(0.01)

yticks = [0.2, 0.4, 0.6, 0.8, 1.0]
ax.set_yticks(yticks)
ax.set_yticklabels([])
ax.set_rlabel_position(0)

ax.yaxis.grid(True, linestyle='--', color='lightgray', zorder=1)
ax.xaxis.grid(False)
ax.set_xticks([])
ax.spines['polar'].set_visible(False)

theta_circle = np.linspace(0, 2 * np.pi, 200)
ax.plot(theta_circle, np.zeros_like(theta_circle), color='black', linewidth=1.5, zorder=4)

N_groups = len(groups)
N_bars_per_group = len(features)

group_width = 2 * np.pi / N_groups
gap_between_groups = group_width * 0.18
usable_width = group_width - gap_between_groups
bar_width = (usable_width / N_bars_per_group) * 0.9

for i in range(N_groups):
    start_angle = i * group_width + gap_between_groups / 2

    for j in range(N_bars_per_group):
        bar_angle = start_angle + j * (usable_width / N_bars_per_group) + (usable_width / N_bars_per_group) / 2
        ax.bar(bar_angle, plot_values[i][j], width=bar_width, bottom=0,
               color=colors[i], edgecolor='white', linewidth=0.5, zorder=3)
        disp_angle = bar_angle * -1 + np.pi / 2
        rot_deg = np.degrees(disp_angle) % 360
        if 90 < rot_deg <= 270:
            ha = 'right'
            text_rot = rot_deg - 180
        else:
            ha = 'left'
            text_rot = rot_deg
        label_text = features[j]
        ax.text(bar_angle, plot_values[i][j] + 0.05, label_text,
                ha=ha, va='center',
                rotation=text_rot, fontsize=30, fontname='Arial',
                rotation_mode='anchor',
                zorder=5)


fig2, ax2 = plt.subplots(figsize=(6, 3),dpi=600)
ax2.axis("off")

legend_elements = [Patch(facecolor=colors[i], label=groups[i]) for i in range(N_groups)]
leg = ax2.legend(handles=legend_elements, loc="center", frameon=True,
                 edgecolor='gray', prop={'family': 'Arial', 'size': 13},
                 handlelength=1.8, handleheight=1.8)
leg.get_frame().set_linewidth(1.2)
plt.tight_layout()
plt.savefig("a.png", dpi=900, bbox_inches="tight")


# plt.savefig('../../pic/radar/Bold_Radar.png', bbox_inches='tight', dpi = 900)
plt.show()