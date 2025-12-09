
# Script to take the main evaluation measures CSV table, and produce legible plot figures.

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import seaborn as sns
sns.set_theme(style="whitegrid")

from tools import utils



# Load the data table
df = pd.read_csv("tbl_embeddings_quant_eval.csv")


for plotstat in ['precision_at_1', 'circvar']:
    # start a new pdf
    pdf = PdfPages(f"plots/plot_evalmeasures_{plotstat}.pdf")
    
    for dataset in [
        'cifar10',
        'cifar100',
        ]:
        for quantmode in ['none', '8bit', '1bit',
            'pq8_2', 'pq8_1',
            'pq4_4', 'pq4_2',
                ]:
            subset = df.loc[(df['dataset']==dataset) & (df['quantmode']==quantmode)]
                        
            # a plot with multiple lines: x axis has dimnality AND koleo, y axis has the measure
            g = sns.catplot(
                data=subset, x="koleo", y=plotstat,
                hue="projmode", palette="YlGnBu_d",
                hue_order=['sphere', 'torusC', 'torusN'],
                markers=['o', 's', '^'], linestyles=['-', '--', '-'],
                col="projection_dim",
                capsize=.2,
                kind="point", height=6, aspect=.25,
                linewidth=1, markersize=8,
            )
            #sns.move_legend(g, loc='lower right', bbox_to_anchor=g.axes_dict[128].bbox)
            g.despine(left=True)
            # Iterate over each subplot to customize further
            for whichsubplot, (projdim, ax) in enumerate(g.axes_dict.items()):
                if whichsubplot==0:
                    ax.set_title(f"        {dataset}, quant {quantmode}")
                else:
                    ax.set_title("")
                ax.set_xlabel(f"{projdim}D")
                ax.set_ylim(0, 1)
                
            
            pdf.savefig()
            plt.close()
            del g

    pdf.close()
