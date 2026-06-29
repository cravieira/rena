#!/usr/bin/python3

'''
Draw the design space exploration (DSE) plot of Rena accelerators. Requires a 'rn-dse.csv' file in the same directory.
'''

import pandas as pd
import common
import plot_common
import numpy as np
import matplotlib as mpl
#mpl.style.use('classic')

def preprocess_df(df):
    """Add extra information to the dataframe based on models names"""
    df = common.append_dim(df)
    #df = common.append_class(df)
    df['area_efficiency'] = 1/df['clb']
    return df

# Find pareto points. Based on:
# https://stackoverflow.com/a/40239615
def is_pareto_efficient(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]>c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient

def main():
    df = pd.read_csv('rn-dse.csv')
    df = preprocess_df(df)
    df['latency_time'] = df['latency'] * df['achieved_cp'] # In nS
    df['pareto'] = 'No'
    short_df = df.loc[:, ['name', 'dim', 'lut', 'ff', 'clb', 'latency_time', 'throughput', 'efficiency', 'area_efficiency', 'power', 'energy', 'edp', 'pareto']]

    # Plot
    plot = dict()
    cm = 1/2.54  # centimeters in inches
    IEEE_column_width = 8.89*cm # Column width in IEEE paper in cms
    #plot_common.make_plot(plot, path=None)
    adjustment = 1.5
    #plot['set_size_inches'] = {'w': IEEE_column_width*adjustment, 'h': 5*cm*adjustment}
    plot['set_size_inches'] = {'w': IEEE_column_width*adjustment, 'h': IEEE_column_width*adjustment}

    axes = []
    plot['axes'] = axes
    ax = dict()
    axes.append(ax)
    ax['xlabel'] = 'Throughput (classifications/s)'
    ax['ylabel'] = 'Area Efficiency (1/CLBs)'
    ax['ticklabel_format'] = {'scilimits': (0,0), 'useMathText': True}
    ax['grid'] = {'axis': 'both', 'linestyle': '--'}
    ax['loc'] = (0, 0)
    ax['legend'] = {}
    #ax['set_ylim'] = {'top': 0.0003}

    curves = dict()
    ax['curves'] = curves

    def find_pareto_points(df):
        pareto_keys = ['throughput','area_efficiency']
        df_costs = np.array(df[pareto_keys])
        efficient_ind = is_pareto_efficient(df_costs) # Use this array to index the input dataframe for the pareto points
        pareto_points = df_costs[efficient_ind].T
        ind = np.argsort(pareto_points[0])
        pareto_points = pareto_points[:, ind]
        return pareto_points, efficient_ind
    all_pareto, pareto_ind = find_pareto_points(df)

    # Print latex table of best results
    # TODO: Create a new column to inform if the point is pareto and add them to the table
    df['pareto'] = 'No'
    df.loc[pareto_ind, 'pareto'] = 'Yes'
    print(df)


    min_lut = df.loc[df['lut'].idxmin()].to_frame().T
    min_ff = df.loc[df['ff'].idxmin()].to_frame().T
    min_clb = df.loc[df['clb'].idxmin()].to_frame().T
    min_power = df.loc[df['power'].idxmin()].to_frame().T
    min_energy = df.loc[df['energy'].idxmin()].to_frame().T
    min_edp = df.loc[df['edp'].idxmin()].to_frame().T

    best_throughput = df.loc[df['throughput'].idxmax()].to_frame().T
    best_latency_time = df.loc[df['latency_time'].idxmin()].to_frame().T
    df_pareto_points = df.loc[df['pareto'] == 'Yes']

    print("min lut:", min_lut)
    print("min ff:", min_ff)
    print("min clb:", min_clb)
    print("min power:", min_power)
    print("min energy:", min_energy)
    print("min edp:", min_edp)
    print("best_throughput", best_throughput)
    print("best_latency_time", best_latency_time)
    print("all pareto points", df_pareto_points)

    all_color = 'C0'
    curves['all_frontier'] = {'type': 'plot', 'x': all_pareto[0], 'y': all_pareto[1], 'linestyle': '--', 'color': all_color}

    markersize = 80
    zorder = 2.5 # Draw scatter plots on top of the lines
    curves['all_accelerators'] = {'type': 'scatter', 'x': df['throughput'], 'y': df['area_efficiency'], 's': markersize, 'label': 'Design Points', 'marker': '*', 'edgecolor': 'black', 'zorder': zorder+0.2, 'color': all_color}

    curves['best_energy'] = {'type': 'scatter', 'x': min_energy['throughput'], 'y': min_energy['area_efficiency'], 's': markersize*1, 'label': 'Best Energy', 'marker': 'o', 'edgecolor': 'black', 'zorder': zorder+0.2, 'color': 'green'}
    curves['best_edp'] = {'type': 'scatter', 'x': min_edp['throughput'], 'y': min_edp['area_efficiency'], 's': markersize*1, 'label': 'Best EDP', 'marker': 'o', 'edgecolor': 'black', 'zorder': zorder+0.2, 'color': 'purple'}
    curves['best_power'] = {'type': 'scatter', 'x': min_power['throughput'], 'y': min_power['area_efficiency'], 's': markersize*1, 'label': 'Best Power', 'marker': 'o', 'edgecolor': 'black', 'zorder': zorder+0.2, 'color': 'red'}

    plot_common.make_plot(plot, path='_plots/dse.pdf')


    latex_df = pd.concat([min_lut, min_ff, min_clb, best_latency_time, best_throughput, min_power, min_energy, min_edp, df_pareto_points])
    latex_df = latex_df.drop_duplicates() # Remove duplicate rows
    # Simplify names. Instead of the full name, use a short dp<>-cp<> combination
    latex_df['name'] = latex_df['name'].str.replace('vitis_dse_rn-d1536-seg_size128-', '', regex=True)
    # Convert latency_time from nS to uS
    latex_df['latency_time'] = latex_df['latency_time'] * 10**-3
    # Convert energy from Joules to uJ
    latex_df['energy'] = latex_df['energy'] * 10**6
    # Convert EDP from J*s to J*s*10^-12 by scaling the results
    latex_df['edp'] = latex_df['edp'] * 10**12


    columns=['name', 'lut', 'ff', 'clb', 'latency_time', 'throughput', 'power', 'energy', 'edp', 'pareto']
    latex_df = latex_df[columns] # Reorder columns order
    col_fmt = 'c' * len(columns)
    table_str = latex_df.to_latex(
            None,
            columns=columns,
            header=['Name', 'LUT',  'FF', 'CLB', r'Latency ($\mu$s)', 'Throughput (Class./s)', 'Power (W)', r'Energy ($\mu$J)', r'EDP (Js$\times10^{{-12}}$)', 'Pareto'],
            column_format=col_fmt,
            float_format="%.2f",
            index=False,
            caption="Summary of FPGA implementation results. The best result for each column is highlighted.",
            )
    print(table_str)

if __name__ == '__main__':
    main()

