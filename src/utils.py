import matplotlib.pyplot as plt

def plotSeries(
    series,
    kind="bar",
    title=None,
    xlabel=None,
    ylabel=None,
    rotation=45,
    color="steelblue",
    top_n=None,
    figsize=(10, 5),
    save_path=None,
):
    """
    Plot any pandas Series (e.g. variation, mean, correlation).

    Args:
        series (pd.Series): data to plot (index = feature names).
        kind (str): plot type, e.g. "bar", "line".
        title (str): figure title.
        xlabel (str): label for x-axis.
        ylabel (str): label for y-axis.
        rotation (int): rotation of x-ticks.
        color (str): bar/line color.
        top_n (int): show only top N items.
        figsize (tuple): figure size.
        save_path (str): if set, saves the figure to this path.
    """
    if top_n:
        series = series.sort_values(ascending=False).head(top_n)

    plt.figure(figsize=figsize)
    series.plot(kind=kind, color=color)
    plt.title(title if title else "")
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()
    return series
