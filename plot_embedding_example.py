
# plot_embedding_example

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
sns.set_theme(style="whitegrid")

datadir = "embeddings"
plotsdir = "plots"


def decode_koleo_val(astr):
    "Supply the filename-encoded koleo weight, and this will return it as a number"
    if "e" not in astr:
        astr = astr.replace("-", ".")
    return np.array(astr, dtype=float)


for ndims in [16, 128]:
    unscaler = np.sqrt(ndims/2.0)  # undo the total-L2-norm operation applied, for simplifed visualisation
    xdim = 7
    ydim = 4

    for koleo in ['0', '0-001', '0-1', '1']:
        df = pd.read_csv(f"{datadir}/supcon_first_stage_cifar100_D{ndims}_torusN_koleo{koleo}_swa_val_embed.csv", header=None)
        df = df * unscaler
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        sns.scatterplot(x=xdim, y=ydim,
                        alpha=0.3, linewidths=0,
                        data=df, ax=ax)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"Dimension {xdim}")
        ax.set_ylabel(f"Dimension {ydim}")
        ax.set_title(f"KoLeo {decode_koleo_val(koleo)}")
        #   Save as PNG (because PDF scatter of 10k points is too heavy to render nicely for everyone)
        fig.savefig(f"{plotsdir}/plot_2Dembeddingexample_D{ndims}_koleo{koleo}.png")

