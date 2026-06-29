DESCRIPTION="Rena plots"

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from natsort import humansorted

import plot_common

PLOT_DIR='_plots'

CM = 1/2.54  # centimeters in inches
IEEE_column_width = 8.89*CM # Column width in IEEE paper in cms

def label_id(row):
    """
    Create labels for each experiment type based on the experiment's parameters
    """
    if row['vsa'] == 'CGR':
        # Some CSV files created do not have a column for bundle mode as I
        # hadn't thought of adding them at the time I performed the experiment
        bundle_type = 'mode'
        if 'cgr_bundle' in row:
            if row['cgr_bundle'] is not None:
                bundle_type = row['cgr_bundle']
        block_size = int(row['cgr_block_size'])
        vsa_class = f"{row['vsa']}{block_size}-bundle_{bundle_type}"
    else:
        vsa_class = row['vsa']

    return vsa_class

def ax_style_common(ax):
    """Set attributes that are commons between accuracy and iteration plots"""
    ax['loc'] = (0, 0)
    ax['grid'] = {'axis': 'both', 'linestyle': '-'}
    ax['legend'] = {}
    ax['legend']['loc'] = 'best'
    #ax['legend']['bbox_to_anchor'] = (1.05, 1)
    ax['spines'] = {'top': False, 'right': False}

    # X axis - Problem size
    ax['xlabel'] = 'Problem Size, $M^3$'
    ax['set_xscale'] = {'value': 'log'}
    ax['set_xlim'] = {'left': 10**6, 'right': 10**11}

    return ax

def ax_style_accuracy(ax):
    """Set plot style for accuracy plots"""
    ax = ax_style_common(ax)
    # Y axis - Accuracy
    ax['ylabel'] = 'Factorization Accuracy'
    ax['set_ylim'] = {'bottom': 0.0, 'top': 1.0*1.01}
    #ax['legend']['loc'] = 'lower right'
    return ax

def ax_style_iteration(ax, max_iter=False):
    """Set plot style for iteration plots"""
    ax = ax_style_common(ax)
    # Y axis - Iterations
    ax['ylabel'] = 'Number of iterations'
    ax['set_yscale'] = {'value': 'log'}
    ax['set_ylim'] = {'bottom': 10**0, 'top': 10**7}

    # Draw a line to indicate the ideal max number of iterations and a polygon
    # above it to hide the grid over the plot. The idea is to indicate the
    # range that RN is good, i.e., below the line
    if max_iter is not None:
        search_space = [10**4, 10**11]
        S = np.array(search_space)
        F = 3
        # Rearrengement of Equation 2 in the paper "In-memory factorization of
        # holographic perceptual representations" to compute the max number of
        # iterations based on the search space and the number of factors
        max_iter = S**((F-1)/F)/F
        # Place max_iter line on top regardless
        zorder=2.999
        ax['curves']['max_iter_line'] = {'type': 'plot', 'x': [S[0], S[1]], 'y': [max_iter[0], max_iter[-1]], 'label': None, 'color': 'black', 'linestyle': 'dashed', 'zorder': zorder}

        xy = [
                [10**4, max_iter[0]], # bottom left
                [10**4, 10**7], # top left
                [10**11, 10**7], # top right
                [10**11, max_iter[1]] # bottom right
        ]
        ax['curves']['hide_top_area'] = {'type': 'Polygon', 'xy': xy, 'label': None, 'color': 'white'}
        ax['set_axisbelow'] = True
    return ax

def parse_csv(csv_path):
    """
    Parse the reports dataframe in a csv file

    Also convert missing values to Python "None"
    """
    df = pd.read_csv(csv_path)
    df = df.where(df.notnull(), None) # Convert missing values to None
    return df

def get_factors(df: pd.DataFrame):
    """
    Find the number of factors in the experiments

    Raises exception if the dataframe contains reports with different factors.
    Each dataframe/csv file must report results for a single factor only to
    avoid mistakes.
    """

    factors = np.unique(df['factors'].to_numpy())
    if factors.ndim > 1:
        raise RuntimeError(f'Found uneven number of factors in dataframe! Factors = {factors}')

    return factors

def get_codebook_sizes(df: pd.DataFrame):
    """docstring for get_codebook_sizes"""
    cb_sizes = np.unique(df['codebook_size'].to_numpy())
    return cb_sizes

def get_search_space(df):
    """
    Compute search space from CB sizes and factors
    """
    factors = get_factors(df)
    cb_sizes = get_codebook_sizes(df)
    search_space = cb_sizes**factors
    return search_space

def get_max_iter(df):
    """
    Find the ideal max number of iterations for each experiment

    The ideal max number of iterations is computed based on the codebook size
    and the number of features.
    """
    codebook_size = get_codebook_sizes(df)
    factors = get_factors(df)
    iter_fac = 1
    max_iter = (codebook_size**(factors - 1)) / factors * iter_fac
    return max_iter

def interpolate(x, y, npoints=1000):
    """
    Interpolate X and Y data and return new x_interp/y_interp arrays with
    npoints.
    """
    x_interp = np.geomspace(x[0], x[-1], npoints)
    y_interp = np.interp(x_interp, x, y)

    return x_interp, y_interp

def _segment_ref(x, y, y_ref, threshold):
    """
    Segment arrays x and y in intervals greater and lesser than threshold
    applied to y_ref.
    """
    y_idx = y_ref < threshold

    cuts = np.flatnonzero(np.diff(y_idx))
    cuts = np.hstack([0, cuts+1, len(y_ref)])

    x_above = []
    y_above = []
    x_below = []
    y_below = []
    for i in range(1, cuts.size):
        if y_ref[cuts[i]-1] < threshold:
            x_below.append(x[cuts[i-1]:cuts[i]])
            y_below.append(y[cuts[i-1]:cuts[i]])
        else:
            x_above.append(x[cuts[i-1]:cuts[i]])
            y_above.append(y[cuts[i-1]:cuts[i]])

    return [{'x': x_above, 'y': y_above}, {'x': x_below, 'y': y_below}]

def segment_acc(x, y_acc, npoints=1000, threshold=0.99):
    """
    Create 2 segments of arrays x and y based on threshold
    """
    x_range, y_acc_i = interpolate(x, y_acc, npoints=npoints)
    segments = _segment_ref(x_range, y_acc_i, y_acc_i, threshold)

    return segments

def segment_iter(x, y_iter, y_acc, npoints=1000, threshold=0.99):
    """docstring for segment_iter"""
    x_range, y_acc_i = interpolate(x, y_acc, npoints=npoints)
    _, y_iter_i = interpolate(x, y_iter, npoints=npoints)
    segments = _segment_ref(x_range, y_iter_i, y_acc_i, threshold)

    return segments

def add_segment_plot(curves: dict, segments, *, label: str, color: str, **kwargs):
    """Add segments created with segment_array() to plots"""
    above, below = segments
    for i in range(len(above['x'])):
        curves[f'{label}_above{i}'] = {'type': 'plot', 'x': above['x'][i], 'y': above['y'][i], 'label': label, 'color': color, **kwargs}

    for i in range(len(below['x'])):
        curves[f'{label}_below{i}'] = {'type': 'plot', 'x': below['x'][i], 'y': below['y'][i], 'label': None, 'color': color, 'linestyle': 'dotted', **kwargs}


    if below['x']:
        # Add marker to indicate the end of the last 99% accuracy segment if there
        # is no recover, i.e., there is no >99% region after a below
        x_above_last_end = above['x'][-1][-1] # End of the last above region
        x_below_last_begin = below['x'][-1][0] # Begin of the last below region
        if x_above_last_end < x_below_last_begin:
            # Add marker to indicate the end of the last 99% accuracy segment
            x = x_above_last_end
            y = above['y'][-1][-1]

            zorder = kwargs.get('zorder', None)
            # Place the marker above all plot lines if the line should be drawed with zorder
            if zorder:
                zorder = zorder+0.00001+1
            curves[f'{label}_marker'] = {'type': 'plot', 'x': x, 'y': y, 'color': color, 'marker': 'X', 'markerfacecolor': 'white', 'zorder': zorder}

    return curves

def noise_comparison(csv_path_dict):
    '''
    Compare the performance of an RN using normal and PRNG noise injection.
    '''
    df_dict = {}
    for k, v in csv_path_dict.items():
        df = parse_csv(v)
        search_space = get_search_space(df)

        d = {
                'df': df,
                'search_space': search_space
        }
        df_dict[k] = d


    acc_dict = {}
    avg_iter_dict = {}

    # Create unique ID labels based on the experiments' paramaters
    # Baseline -
    #df = df_dict['baseline']['df']
    #df_short = df[['codebook_size', 'acc_factors', 'acc_frames', 'niter_avg']]
    #df_short = df_short.groupby(['codebook_size']).mean()
    #df_dict['baseline'].update(df_short.items())

    def add_plot_data(df_short, search_space, acc_dict, avg_iter_dict):
        for group_name, group_data in df_short.groupby(['id']):
            acc_dict[group_name[0]] = {
                    'x': search_space,
                    'y': group_data['acc_factors'].to_numpy(),
                    }
            avg_iter_dict[group_name[0]] = {
                    'x': search_space,
                    'y': group_data['niter_avg'].to_numpy(),
                    }
        return acc_dict, avg_iter_dict

    ## Normal noise
    #def label_normal(row):
    #    """
    #    Create labels for each experiment type based on the experiment's parameters
    #    """
    #    return f"normal-{row['normal_std']}"

    #df = df_dict['normal']['df']
    #df['id'] = df.apply(label_normal, axis=1)
    #df_short = df[['id', 'codebook_size', 'acc_factors', 'acc_frames', 'niter_avg']]
    #df_short = df_short.groupby(['id', 'codebook_size']).mean()
    #df_dict['normal']['mean_df'] = df_short
    #acc_dict, avg_iter_dict = add_plot_data(df_short, df_dict['normal']['search_space'], acc_dict, avg_iter_dict)

    # Xorshift noise
    def label_xorshift(row):
        """docstring for label_prng"""
        return f"xorshift-{row['xorshift_max']}"

    df = df_dict['xorshift']['df']
    df['id'] = df.apply(label_xorshift, axis=1)
    df_short = df[['id', 'codebook_size', 'acc_factors', 'acc_frames', 'niter_avg']]
    df_short = df_short.groupby(['id', 'codebook_size']).mean()
    #df_dict['xorshift'].update(df_short.items())
    acc_dict, avg_iter_dict = add_plot_data(df_short, df_dict['xorshift']['search_space'], acc_dict, avg_iter_dict)

    # Filter out some data to avoid plot pollution
    keys = ['xorshift-2']
    for k in keys:
        acc_dict.pop(k)
        avg_iter_dict.pop(k)

    # Prepare plot data
    ## Normal noise
    npoints = 3000

    curves_acc = {}
    curves_iter = {}

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    color_it = iter(colors)

    # Order dictionary keys to they display nice in the plots
    acc_dict = dict(humansorted(acc_dict.items()))
    # Select curves' zorder
    zorder = {
            'xorshift-2': 5,
            'xorshift-4': 3,
            'xorshift-8': 4,
            'xorshift-16': 2,
            'xorshift-32': 1
    }

    def _apply_zorder(d: dict, zorder: dict):
        for k in d.keys():
            if k in zorder:
                d[k]['zorder'] = zorder[k]
        return d
    acc_dict = _apply_zorder(acc_dict, zorder)
    avg_iter_dict = _apply_zorder(avg_iter_dict, zorder)

    for k, v in acc_dict.items():
        acc = v['y']
        acc_segments = segment_acc(v['x'], acc, npoints=npoints)

        niter_avg_segments = segment_iter(v['x'], avg_iter_dict[k]['y'], acc, npoints=npoints)

        c = next(color_it)
        zorder=v.get('zorder', None)
        if zorder:
            zorder = 2+zorder/100 # Set zorder as 2.<zorder_val> as suggested in matplotlib documentation
        curves_acc = add_segment_plot(curves_acc, acc_segments, label=k, color=c, zorder=zorder)
        curves_iter = add_segment_plot(curves_iter, niter_avg_segments, label=k, color=c, zorder=zorder)

    # Plot #
    # Plot accuracy
    plot = dict()
    adjustment = 1.8
    plot_size = {'w': IEEE_column_width*adjustment, 'h': 4*CM*adjustment}
    plot['set_size_inches'] = plot_size

    axes = []
    plot['axes'] = axes
    ax = dict()
    axes.append(ax)
    ax = ax_style_accuracy(ax)
    x_lim = {'left': 10**6, 'right': 10**11} # Adjust problem size
    ax['set_xlim'] = x_lim

    curves = dict()
    ax['curves'] = curves_acc

    plot_common.make_plot(plot, path=f'{PLOT_DIR}/rena-noise-comparison-accuracy.pdf')

    # Plot iterations
    plot = dict()
    plot['set_size_inches'] = plot_size

    axes = []
    plot['axes'] = axes
    ax = dict()
    axes.append(ax)

    curves = dict()
    ax['curves'] = curves_iter

    ax = ax_style_iteration(ax, max_iter=True)
    ax['set_xlim'] = x_lim

    plot_common.make_plot(plot, path=f'{PLOT_DIR}/rena-noise-comparison-iteration.pdf')


def sota_comparison(csv_path_dict):
    '''
    Compare the performance of different RN algorithms: baseline, and Rena
    '''
    df_dict = {}
    for k, v in csv_path_dict.items():
        df = parse_csv(v)
        search_space = get_search_space(df)

        d = {
                'df': df,
                'search_space': search_space
        }
        df_dict[k] = d

    acc_dict = {}
    avg_iter_dict = {}
    zorder = {
            'baseline': 5,
            #'baseline-opt': 4,
            'acf': 3,
            'rena': 2,
            'imf': 1
            }

    # Create unique ID labels based on the experiments' paramaters
    # Baseline
    df = df_dict['baseline']['df']
    df_short = df[['codebook_size', 'acc_factors', 'acc_frames', 'niter_avg']]
    df_short = df_short.groupby(['codebook_size']).mean()
    #df_dict['baseline'].update(df_short.items())

    # Notice the renaming from "baseline" to "Baseline"
    acc_dict['Baseline'] = {
            'x': df_dict['baseline']['search_space'],
            'y': df_short['acc_factors'].to_numpy(),
            'zorder': zorder['baseline']
            }
    avg_iter_dict['Baseline'] = {
            'x': df_dict['baseline']['search_space'],
            'y': df_short['niter_avg'].to_numpy(),
            'zorder': zorder['baseline']
            }

    # Optimized baseline
    #df = df_dict['baseline-opt']['df']
    #df_short = df[['codebook_size', 'acc_factors', 'acc_frames', 'niter_avg']]
    #df_short = df_short.groupby(['codebook_size']).mean()

    # Notice the renaming from "baseline-opt" to "Baseline+Opt"
    #acc_dict['Baseline+Opt'] = {
    #        'x': df_dict['baseline-opt']['search_space'],
    #        'y': df_short['acc_factors'].to_numpy(),
    #        'zorder': zorder['baseline-opt']
    #        }
    #avg_iter_dict['Baseline+Opt'] = {
    #        'x': df_dict['baseline-opt']['search_space'],
    #        'y': df_short['niter_avg'].to_numpy(),
    #        'zorder': zorder['baseline-opt']
    #        }

    # IBM Asymmetric Codebook NeurIPS workshop paper.
    # Data available at: https://github.com/IBM/in-memory-factorizer/blob/main/experiments/300a_assymetric_cb/f3_optimum.csv
    ACF_accuracy = [1, 1, 1, 1, 1, 1, 0.993815104166667, 0.9892578125, 1, 1, 0.998046875, 0.9990234375, 0.994140625, 0.989583333333333, 0.9931640625, 0.9990234375, 1, 1, 1, 1, 1, 1, 1, 1, 0.9970703125]
    ACF_iter = [1, 2, 2, 2, 2.0615234375, 2.2529296875, 2.689453125, 3.796875, 7.4287109375, 11.6875, 20.38671875, 27.1376953125, 47.0517578125, 80.9814453125, 97.9951171875, 166.8984375, 154.8837890625, 177.6640625, 283.5380859375, 438.55078125, 706.3671875, 1354.697265625, 2858.4052734375, 7572.9951171875, 27560.5458984375]
    ACF_problem_size = [8, 27, 64, 125, 216, 512, 1000, 2197, 4913, 10648, 21952, 46656, 97336, 216000, 456533, 1000000, 2146689, 4657463, 9938375, 21484952, 46268279, 99897344, 214921799, 463684824, 1000000000]
    acc_dict['ACF'] = {
            'x': ACF_problem_size,
            'y': ACF_accuracy,
            'zorder': zorder['acf']
            }
    avg_iter_dict['ACF'] = {
            'x': ACF_problem_size,
            'y': ACF_iter,
            'zorder': zorder['acf']
            }

    # IBM IMF Nature paper - Data obtained form their open data repository
    IBM_accuracy = [0.9921875, 0.99121094, 0.99169922, 0.99283854, 0.99267578, 0.99283854, 0.99267578, 0.99267578, 0.99267578, 0.99235026, 0.9921875, 0.99267578, 0.9921875, 0.9921875, 0.9921875, 0.9921875, 0.9921875, 0.9921875, 0.9921875, 0.99023438]
    IBM_problem_size = [10648, 21952, 46656, 97336, 216000, 456533, 1000000, 2146689, 4657463, 9938375, 21484952, 46268279, 99897344, 214921799, 463684824, 1000000000, 2156689088, 4640749632, 9993948264, 100000000000]
    IBM_iter = [8.256, 12.887, 17.148, 26.522, 38.948, 50.454, 74.766, 111.827, 176.784, 278.984, 537.792, 846.475, 1416.166, 2401.84, 4551.632, 7776.545, 14002.222, 23923.03, 45894.028, 338457.5]
    acc_dict['IMF'] = {
            'x': IBM_problem_size,
            'y': IBM_accuracy,
            'zorder': zorder['imf']
            }
    avg_iter_dict['IMF'] = {
            'x': IBM_problem_size,
            'y': IBM_iter,
            'zorder': zorder['imf']
            }

    def add_plot_data(df_short, search_space, acc_dict, avg_iter_dict):
        for group_name, group_data in df_short.groupby(['id']):
            acc_dict[group_name[0]] = {
                    'x': search_space,
                    'y': group_data['acc_factors'].to_numpy(),
                    }
            avg_iter_dict[group_name[0]] = {
                    'x': search_space,
                    'y': group_data['niter_avg'].to_numpy(),
                    }
        return acc_dict, avg_iter_dict

    # Xorshift noise
    def label_xorshift(row):
        """docstring for label_prng"""
        return f"xorshift-{row['xorshift_max']}"

    df = df_dict['xorshift']['df']
    df['id'] = df.apply(label_xorshift, axis=1)
    df_short = df[['id', 'codebook_size', 'acc_factors', 'acc_frames', 'niter_avg']]
    df_short = df_short.groupby(['id', 'codebook_size']).mean()
    acc_dict, avg_iter_dict = add_plot_data(df_short, df_dict['xorshift']['search_space'], acc_dict, avg_iter_dict)

    # Filter out some data to avoid plot pollution
    keys = ['xorshift-2', 'xorshift-4', 'xorshift-16', 'xorshift-32']
    for k in keys:
        acc_dict.pop(k)
        avg_iter_dict.pop(k)

    # Rename xorshift-8 to Rena since we that it should be the default Rena configuration now as it works better
    acc_dict['Rena'] = acc_dict.pop('xorshift-8')
    acc_dict['Rena']['zorder'] = zorder['rena']
    avg_iter_dict['Rena'] = avg_iter_dict.pop('xorshift-8')
    avg_iter_dict['Rena']['zorder'] = zorder['rena']
    print(f'Rena avg_iter: {avg_iter_dict["Rena"]}')

    # Prepare plot data
    ## Normal noise
    npoints = 3000

    curves_acc = {}
    curves_iter = {}

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    color_it = iter(colors)
    for k, v in acc_dict.items():
        acc = v['y']
        acc_segments = segment_acc(v['x'], acc, npoints=npoints)

        niter_avg_segments = segment_iter(v['x'], avg_iter_dict[k]['y'], acc, npoints=npoints)

        c = next(color_it)
        zorder=v.get('zorder', None)
        if zorder:
            zorder = 2+zorder/100 # Set zorder as 2.<zorder_val> as suggested in matplotlib documentation
        curves_acc = add_segment_plot(curves_acc, acc_segments, label=k, color=c, zorder=zorder)
        curves_iter = add_segment_plot(curves_iter, niter_avg_segments, label=k, color=c, zorder=zorder)

    # Plot #
    AX_LEFT = 10**6
    # Plot accuracy
    plot = dict()
    adjustment = 1.8
    plot_size = {'w': IEEE_column_width*adjustment, 'h': 4*CM*adjustment}
    plot['set_size_inches'] = plot_size

    axes = []
    plot['axes'] = axes
    ax = dict()
    axes.append(ax)
    ax = ax_style_accuracy(ax)
    ax['set_xlim'] = {'left': AX_LEFT, 'right': 10**11} # Adjust problem size

    curves = dict()
    ax['curves'] = curves_acc

    plot_common.make_plot(plot, path=f'{PLOT_DIR}/rena-sota-comparison-accuracy.pdf')

    # Plot iterations
    plot = dict()
    plot['set_size_inches'] = plot_size

    axes = []
    plot['axes'] = axes
    ax = dict()
    axes.append(ax)

    curves = dict()
    ax['curves'] = curves_iter

    ax = ax_style_iteration(ax, max_iter=True)
    ax['set_xlim'] = {'left': AX_LEFT, 'right': 10**11} # Adjust problem size

    plot_common.make_plot(plot, path=f'{PLOT_DIR}/rena-sota-comparison-iteration.pdf')

if __name__ == "__main__":
    # Plot noise comparison experiment - Compare normal noise with xorshift noise injection
    noise_comparison({
        'xorshift': 'rn-xorshiftsweep-fpl.csv'
        }
    )

    # Compare our proposed xorshift with the SOTA RN accelerators
    sota_comparison({
        'baseline': 'rn-baseline-fpl.csv',
        'xorshift': 'rn-xorshiftsweep-fpl.csv'
        }
    )

