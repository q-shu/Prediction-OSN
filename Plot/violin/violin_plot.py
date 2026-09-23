import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from itertools import cycle


def violin_plot(group_name, pic_name, fill_col, edge_col):
    df_raw = pd.read_excel('../../data/data.xlsx')
    df_all = df_raw.melt(var_name='Group', value_name='Value')

    target_group = group_name
    df = df_all[df_all['Group'] == target_group].dropna()

    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.weight'] = 'normal'
    plt.rcParams['axes.labelweight'] = 'normal'
    plt.rcParams['axes.titleweight'] = 'normal'

    fig, ax = plt.subplots(figsize=(3, 4.5),dpi = 600)

    sns.violinplot(
        data=df, x='Group', y='Value',
        palette=[fill_col], hue='Group', legend=False,
        inner='box', linewidth=1.5, ax=ax
    )

    sns.stripplot(
        data=df, x='Group', y='Value',
        palette=[fill_col], hue='Group', legend=False,
        edgecolor=edge_col, linewidth=1.2,
        alpha=0.7, jitter=0.08, size=6, zorder=3, ax=ax
    )

    ax.set_xlabel('')
    ax.set_ylabel('')

    y_min, y_max = df['Value'].min(), df['Value'].max()
    ax.set_ylim(y_min - (y_max - y_min) * 0.1, y_max + (y_max - y_min) * 0.1)

    ax.set_axisbelow(True)
    ax.grid(True, linestyle=(0, (8, 4)), color='gray', linewidth=1.2, alpha=0.6)

    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        spine.set_color('black')

    ax.tick_params(
        direction='in', length=6, width=1.5, labelsize=18,
        bottom=True, top=True, left=True, right=True,pad=10
    )
    ax.tick_params(axis='x', labelsize=18)

    for collection in ax.collections:
        if isinstance(collection, plt.matplotlib.collections.PolyCollection):
            collection.set_edgecolor(edge_col)

    plt.tight_layout()
    save_dir = './V-P/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # plt.savefig(f'{save_dir}{pic_name}.png', bbox_inches='tight', dpi=600)
    plt.show()


if __name__ == '__main__':
    # palette = [
    #     ("#AFDEF3", "#2A3D66"),
    #     ("#F5B7B1", "#943126"),
    #     ("#D7BDE2", "#5B2C6F"),
    #     ("#ABEBC6", "#186A3B"),
    #     ("#F9E79F", "#7D6608"),
    #     ("#D6DBDF", "#2C3E50"),
    #     ("#FAD7A0", "#784212"),
    #     ("#AED6F1", "#1B4F72"),
    #     ("#A9DFBF", "#0B5345"),
    #     ("#A3E4D7", "#0E6655")
    # ]
    palette = [
        ("#AFDEF3", "#2A3D66"),
        ("#A3E4D7", "#0E6655"),
        ("#D7BDE2", "#5B2C6F"),
        ("#ABEBC6", "#186A3B"),
        ("#F9E79F", "#7D6608"),
        ("#D6DBDF", "#2C3E50"),
        ("#FAD7A0", "#784212"),
        ("#AED6F1", "#1B4F72"),
        ("#A9DFBF", "#0B5345"),
        ("#F5B7B1", "#943126"),
        ("#A3E4D7", "#0E6655"),
    ]
    color_cycle = cycle(palette)

    my_data = {
        "Contact angle(°)": 'angle',
        "Pore diameter(nm)": 'diameter',
        "MWCO(Da)" : 'MWCO',
        "RMS roughness(nm)": 'roughness',
        "Film thickness(nm)": 'thickness',
        "Pressure(bar)": 'Pressure',
        "Temperature(°C)": 'Temperature',
        "Concentration(mg/L)": 'Concentration',
        "Solute MW (g/mol)": 'Solute',
        "Permeance(LMH/bar)": 'Permeability',
        "Rejection(%)": 'Rejection'
    }
    # my_data = {
    #     "Contact angle(°)": 'angle',
    #     "Pore diameter(nm)": 'diameter',
    #     "MWCO(Da)" : 'MWCO',
    #     "RMS roughness(nm)": 'roughness',
    #     "Film thickness(nm)": 'thickness',
    #     "Pressure(bar)": 'Pressure',
    #     "Temperature(°C)": 'Temperature',
    #     "Concentration(mg/L)": 'Concentration',
    #     "Solute MW (g/mol)": 'Solute',
    #     "Rejection(%)": 'Rejection',
    # }
    for key, value in my_data.items():
        fill, edge = next(color_cycle)
        violin_plot(key, value, fill, edge)