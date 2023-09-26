import os
from pathlib import Path
import pandas as pd
import time
import numpy as np
import matplotlib as mpl
import os
import matplotlib.pyplot as plt
from datetime import datetime
import dill as pickle
import argparse

from vatic.data.loaders import (
    load_input, RtsLoader, T7kLoader
    )
from vatic.engines import Simulator


# Parameters for running VATIC 
RUC_MIPGAPS = {'RTS-GMLC': 0.01, 'Texas-7k': 0.02}
SCED_HORIZONS = {'RTS-GMLC': 4, 'Texas-7k': 2}


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("date", type=str,
                        help="which days to simulate")

    parser.add_argument("grid", type=str, choices={'RTS-GMLC', 'Texas-7k'},
                        help="which power grid to simulate over")

    parser.add_argument("scen_dir", type=str, help="scenario directory")

    parser.add_argument("init_dir", type=str, help="initial state directory")
    parser.add_argument("ruc_dir", type=str, help="ruc directory")
    parser.add_argument("out_dir", type=str, help="output directory")

    parser.add_argument("nscens", type=int, 
                        help="total number of scenarios")

    parser.add_argument("num_jobs", type=int, 
                        help="number of jobs to be executed")
    
    parser.add_argument("job_partition", type=int, 
                        help="partition of jobs, starting from 0")

    parser.add_argument("reserve_factor", type=float, default=0.05, 
                        help="the marginal operating reserve factor used by "
                             "grids in each training simulation")

    parser.add_argument("load_shed_penalty", type=float, default=1e4,
                        help="load shedding penalty per mwh")

    args = parser.parse_args()

    start_date = pd.to_datetime(args.date, utc=True)
    num_days = 1

    ## get input data
    ruc_file = Path(args.ruc_dir, args.grid.lower() + '-ruc-' + 
        start_date.strftime('%Y-%m-%d') + '.p')
    init_state_in = Path(args.init_dir, args.grid.lower() + 
        '-init-state-' + start_date.strftime('%Y-%m-%d') + '.csv')
    if not init_state_in.exists():
        init_state_in = None
    
    ## get scenario data
    if args.grid == 'RTS-GMLC':
        loader = RtsLoader(init_state_file=init_state_in)
    elif args.grid == 'Texas-7k':
        loader = T7kLoader(init_state_file=init_state_in)
    else:
        raise ValueError("unrecognized grid name")

    sim_scenarios = list(np.array_split(
        range(args.nscens), args.num_jobs)[args.job_partition])

    scen_dfs = loader.load_scenarios(args.scen_dir, [start_date,\
            start_date + pd.Timedelta(1, unit='D')], sim_scenarios)

    for scenario in sim_scenarios:

        gen_data, load_data = loader.create_scenario_timeseries(
            scen_dfs, start_date, start_date + pd.Timedelta(1, unit='D'), scenario)

        siml = Simulator(
                loader.template, gen_data, load_data, None,
                start_date.date(), num_days, solver_options={'Threads': 0}, 
                run_lmps=True, mipgap=RUC_MIPGAPS[args.grid],
                load_shed_penalty = args.load_shed_penalty, 
                reserve_shortfall_penalty = 0,
                reserve_factor=args.reserve_factor, output_detail=3,
                prescient_sced_forecasts=True, ruc_prescience_hour=0,
                ruc_execution_hour=16, ruc_every_hours=24,
                ruc_horizon=48, sced_horizon=SCED_HORIZONS[args.grid],
                hours_in_objective=SCED_HORIZONS[args.grid],
                lmp_shortfall_costs=False,
                init_ruc_file=ruc_file, 
                verbosity=0,
                output_max_decimals=4, create_plots=False,
                renew_costs=None, save_to_csv=False, 
                last_conditions_file=None)
        report_dfs = siml.simulate()

        with open(Path(args.out_dir, 
                'report_dfs-' + str(scenario) + '.p'), 'wb') as handle:
            pickle.dump(report_dfs, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()

    