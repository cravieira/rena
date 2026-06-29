#!/usr/bin/python3

import argparse
import pandas as pd

DESCRIPTION="Merge CSV files"

def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
            'out_csv',
            help="New CSV file with merged data"
            )
    parser.add_argument(
        'csv_files',
        nargs='+',
        help='Input CSV files to be merged'
    )
    args = parser.parse_args()

    dfs = []
    for f in args.csv_files:
        dfs.append(pd.read_csv(f))

    o = pd.concat(dfs)
    o.to_csv(args.out_csv, index=False)
    # TODO:
    # 1: Maybe ensure all factors are equal to avoid mistakes
    #

if __name__ == "__main__":
    main()
