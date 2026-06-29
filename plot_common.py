from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np

PLOT_STYLE='default'
#PLOT_STYLE='bmh'
#PLOT_STYLE='fivethirtyeight'
#PLOT_STYLE='ggplot'
#PLOT_STYLE='seaborn-v0_8'

def _plot(**kwargs):
    path = None
    if 'path' in kwargs:
        path = kwargs['path']
    if path is None:
        plt.show()
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=480)
    plt.close()

def _plt_xscale(xscale=None, **kwargs):
    """docstring for _xscale"""
    if xscale:
        plt.xscale(**xscale)

def _plt_yscale(yscale=None, **kwargs):
    """docstring for _yscale"""
    if yscale:
        plt.yscale(**yscale)

def dict_args(**kwargs):
    """docstring for dict_args"""
    _plt_xscale(**kwargs)
    _plt_yscale(**kwargs)
    _plt_ylabel(**kwargs)
    _plot(**kwargs)

def _title(ax, **kwargs):
    if 'title' in kwargs:
        ax.set_title(kwargs['title'])

def _xticks(**kwargs):
    rotation = None
    if 'xticks_rotation' in kwargs:
        rotation = kwargs['xticks_rotation']
    plt.xticks(rotation=rotation)

def _ylabel(ax, **kwargs):
    if 'ylabel' in kwargs:
        ax.set_ylabel(kwargs['ylabel'])

def _plt_ylabel(**kwargs):
    if 'ylabel' in kwargs:
        plt.ylabel(kwargs['ylabel'])

def _ylim(ax, **kwargs):
    if 'ylim' in kwargs:
        ax.set_ylim(kwargs['ylim'])

# Based on: https://stackoverflow.com/a/49601444
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def simple_bar(x, y, xtick_label=None, ylim=None, path=None):
    fig, ax = plt.subplots()

    x = np.arange(len(x))  # the label locations
    y = y

    ax.bar(x, y)
    plt.xticks(rotation=90)

    if ylim:
        plt.ylim(ylim)

    if xtick_label:
        ax.set_xticks(x, xtick_label)

    plt.tight_layout()
    _plot(path=path)

def boxplot(data, **kwargs):
    fig, ax = plt.subplots()

    labels = None
    if 'labels' in kwargs:
        labels = kwargs['labels']

    plt.grid(axis='y')

    bplot = ax.boxplot(
            data,
            vert=True,
            labels=labels
            )


    _xticks(**kwargs)
    _ylabel(ax, **kwargs)
    _title(ax, **kwargs)

    plt.tight_layout()
    _plot(**kwargs)

def make_plot(plot_dict, path=None, **kwargs):
    """docstring for plot"""
    global PLOT_STYLE
    plt.style.use(PLOT_STYLE)

    subplots_args = plot_dict.get('subplots', {})
    fig, ax = plt.subplots(**subplots_args, layout='constrained')
    #fig.set_size_inches(18.5, 10.5)
    if plot_dict.get('set_size_inches'):
        fig.set_size_inches(**plot_dict['set_size_inches'])

    nrows = subplots_args.get('nrows', 1)
    ncols = subplots_args.get('ncols', 1)
    ax = np.reshape(ax, (nrows, ncols))

    for axis_dict in plot_dict['axes']:
        curves = axis_dict['curves']
        ax_x, ax_y = axis_dict['loc']
        ax_ = ax[ax_x, ax_y]
        for curve_name, curve_data in curves.items():
            d = curve_data.copy()
            plot_type = d['type']
            d.pop('type')
            if plot_type == 'plot':
                x = curve_data['x']
                y = curve_data['y']
                d.pop('x')
                d.pop('y')

                ax_.plot(x, y, **d)
            elif plot_type == 'scatter':
                x = curve_data['x']
                y = curve_data['y']
                d.pop('x')
                d.pop('y')
                ax_.scatter(x, y, **d)

            # Things that aren't exactly curves/plottable data, but shapes
            elif plot_type == 'vline':
                x = d.pop('x')
                ymin = d.pop('ymin')
                ymax = d.pop('ymax')
                ax_.vlines(x, ymin, ymax, **d)
            elif plot_type == 'hline':
                xmin = d.pop('xmin')
                xmax = d.pop('xmax')
                y = d.pop('y')
                ax_.hlines(y, xmin, xmax, **d)
            elif plot_type == 'annotate':
                text = d.pop('text')
                xy = d.pop('xy')
                ax_.annotate(text, xy, **d)

            elif plot_type == 'text':
                x = d.pop('x')
                y = d.pop('y')
                s = d.pop('s')


                tx = None
                if 'coord_system_blend' in d:
                    x_tx_str, y_tx_str = d.pop('coord_system_blend')
                    tx_dict = {
                            'axis': ax_.transAxes,
                            'data': ax_.transData
                            }
                    tx = transforms.blended_transform_factory(tx_dict[x_tx_str], tx_dict[y_tx_str])
                    #if tx_str == 'axis':
                    #    tx = ax_.transAxes
                    #    y = tx(y)
                    #else:
                    #    raise RuntimeError(f"Unsupported transformation \"{tx_str}\" on \"coord_system_y\"")

                ax_.text(x, y, s, **d, transform=tx)

            elif plot_type == 'Polygon':
                xy = d.pop('xy')
                poly = matplotlib.patches.Polygon(xy, **d)
                ax_.add_patch(poly)

        if 'legend' in axis_dict:
            # Do not allow duplicate labels
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax_.legend(by_label.values(), by_label.keys(), **axis_dict['legend'])
            #ax_.legend(**axis_dict['legend'])

        ax_.set_title(axis_dict.get('title', None))
        ax_.set_xlabel(axis_dict.get('xlabel', None))
        ax_.set_ylabel(axis_dict.get('ylabel', None))
        # It is important to set the x/y scales before the limits to avoid bugs
        ax_.set_xscale(**axis_dict.get('set_xscale', {'value': 'linear'}))
        ax_.set_yscale(**axis_dict.get('set_yscale', {'value': 'linear'}))
        ax_.set_xlim(**axis_dict.get('set_xlim', {}))
        ax_.set_ylim(**axis_dict.get('set_ylim', {}))
        ax_.ticklabel_format(**axis_dict.get('ticklabel_format', {}))
        ax_.grid(**axis_dict.get('grid', {}))
        if 'set_axisbelow' in axis_dict:
            ax_.set_axisbelow(axis_dict['set_axisbelow'])

        if 'spines' in axis_dict:
            for spine_name, spine_val in axis_dict['spines'].items():
                ax_.spines[spine_name].set_visible(spine_val)

    dict_args(path=path, **kwargs)
