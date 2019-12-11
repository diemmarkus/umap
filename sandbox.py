import umap
from sklearn.datasets import load_digits

import matplotlib.pyplot as plt

def run_fcs():

    import fmp
    import autogate.utils.vis as fmvis

    filepath = "C:/data/autoflow/daten/vie200/499/499.xml"
    sample = fmp.fcs(filepath)

    events = sample.events()
    mini = events.iloc[0:42]

    embedding = umap.UMAP().fit_transform(mini)

    # show with gt
    labels = sample.gate_labels()
    gatenames = ['intact', 'cd19', 'blast']
    colorset = ['dimgray', 'steelblue', 'red']

    c = fmvis.FmConfig()
    c.result_dir = "C:/temp/img/"
    c.title = "umap"

    ffp = fmvis.FmFcsPlot(events, c=c)
    ffp.gatenames = gatenames
    ffp._colorset = colorset
    ffp.labels = labels.iloc[0:42]
    ffp._data = embedding
    ffp.plot()
    ffp.render()


def run_digits():
    digits = load_digits()

    s = [digits.data[x] for x in range(1,100)]

    # embedding = umap.UMAP().fit_transform(s)
    embedding = umap.UMAP().fit_transform(digits.data)

    cmap = plt.get_cmap('viridis_r')

    for lc in digits.target_names:

        d = embedding[digits.target == lc]

        plt.scatter(
            x=d[:, 0], y=d[:, 1],
            cmap = cmap,
            label=lc,
            # s=s,
            # c=col,
            # alpha=alpha
            # alpha = 0.3
            # marker=m,
            # edgecolors='white', linewidth=0.5,
            # zorder=0 if col == self.bgdcol else 1
            # ax = self.ax()
        )

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


    print("woop")

if __name__ == "__main__":

    run_fcs()