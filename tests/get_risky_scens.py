import os
from pathlib import Path
import pandas as pd
import time
import numpy as np
import argparse
import dill as pickle

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("nscen", type=int,
                        help="total number of scenarios")
    parser.add_argument("alpha", type=float,
                        help="percentage of scenarios considered to be risky")
    parser.add_argument("risk_type", type=str, choices={'daily', 'hourly'},
                        help="type of risk, daily or hourly")
    parser.add_argument("out_dir", type=str, help="output directory")
    parser.add_argument("num_jobs", type=int,
                        help="number of jobs to run reserve scenarios")

    parser.add_argument("load_shed_penalty", type=float, default=1e4,
                        help="load shedding penalty per mwh")
    parser.add_argument("reserve_shortfall_penalty", type=float, default=1e3,
                        help="reserve shortfall penalty per mwh")

    args = parser.parse_args()

    num_risky_scens = int(args.alpha * args.nscen)

    total_costs = {}

    if args.risk_type == 'daily':

        for scen in range(args.nscen):
            df = pd.read_pickle(Path(args.out_dir, 'report_dfs-' + 
                str(scen) + '.p'))['hourly_summary']

            total_costs[scen] = df['FixedCosts'].sum() + df['VariableCosts'].sum() + \
                args.load_shed_penalty * (df['LoadShedding'].sum() + df['OverGeneration'].sum()) + \
                args.reserve_shortfall_penalty * df['ReserveShortfall'].sum()

        risky_scens = dict(sorted(total_costs.items(), 
            key=lambda x:x[1], reverse=True)[0:num_risky_scens])

        alloc_scens = list(risky_scens.keys())

    elif args.risk_type == 'hourly':

        for scen in range(args.nscen):
            df = pd.read_pickle(Path(args.out_dir, 'report_dfs-' + 
                str(scen) + '.p'))['hourly_summary']

            total_costs[scen] = df['FixedCosts'] + df['VariableCosts'] + \
                args.load_shed_penalty * (df['LoadShedding'] + df['OverGeneration']) + \
                args.reserve_shortfall_penalty * df['ReserveShortfall']

        risky_scens = {h: dict(sorted([(scen, total_costs[scen][h])
            for scen in total_costs], key=lambda x:x[1], 
            reverse=True)[0:num_risky_scens]) for h in range(24)
        }

        alloc_scens = list(set.union(*[set(risky_scens[h].keys()) for h in range(24)]))

    scen_chunks = np.array_split(alloc_scens, args.num_jobs)
    job_chunks = {i: list(c) for i, c in enumerate(scen_chunks) if len(c) > 0}

    with open(Path(args.out_dir, args.risk_type + '_cost_alloc_job_chunks.p'), 'wb') as handle:
            pickle.dump(job_chunks, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(Path(args.out_dir, args.risk_type + '_risky_scens.p'), 'wb') as handle:
            pickle.dump(risky_scens, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(Path(args.out_dir, args.risk_type + '_alloc_scens.txt'), 'w') as f:
        for scen in alloc_scens:
            f.write(f"{scen}\n")

if __name__ == '__main__':
    main()