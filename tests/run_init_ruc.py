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
import warnings

from vatic.data.loaders import load_input
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

    parser.add_argument("init_dir", type=str, help="initial state directory")
    parser.add_argument("ruc_dir", type=str, help="ruc directory")
    parser.add_argument("out_dir", type=str, help="output directory")

    parser.add_argument("reserve_factor",
                        type=float, default=0.05,
                        help="the marginal operating reserve factor used by "
                             "grids in each training simulation")
    
    parser.add_argument("load_shed_penalty", type=float, default=1e4,
                        help="load shedding penalty per mwh")
    parser.add_argument("reserve_shortfall_penalty", type=float, default=1e3,
                        help="reserve shortfall penalty per mwh")
    
    parser.add_argument('--risk-type', dest='risk_type', type=str, 
                        choices={'daily', 'hourly'}, help='risk type optional', 
                        required=False)
    
    parser.add_argument('--cost-curve-dir', dest='cost_curve_dir', type=str, 
                        help='cost curve directory optional', required=False)


    args = parser.parse_args()

    start_date = pd.to_datetime(args.date, utc=True)
    num_days = 1

    ruc_file = Path(args.ruc_dir, 
        args.grid.lower() + '-ruc-' + args.date + '.p')

    init_state_in = Path(args.init_dir, args.grid.lower() + 
        '-init-state-' + args.date + '.csv')

    if not init_state_in.exists():
        init_state_in = None
        warnings.warn(f"init state file does not exist "
                      f"using default init state")
    template, gen_data, load_data = load_input(args.grid, args.date, 
        num_days=num_days, init_state_file=init_state_in)

    if args.risk_type != None and args.cost_curve_dir != None:
        cost_curve = Path(args.cost_curve_dir, 
            'cost-curve-' + args.date + '.p')
    else:
        cost_curve = None


    siml = Simulator(
            template, gen_data, load_data, None,
            start_date.date(), num_days, solver_options={'Threads': 0}, 
            run_lmps=False, mipgap=RUC_MIPGAPS[args.grid],
            load_shed_penalty = args.load_shed_penalty, 
            reserve_shortfall_penalty = args.reserve_shortfall_penalty,
            reserve_factor=args.reserve_factor, output_detail=3,
            prescient_sced_forecasts=True, ruc_prescience_hour=0,
            ruc_execution_hour=16, ruc_every_hours=24,
            ruc_horizon=48, sced_horizon=SCED_HORIZONS[args.grid],
            hours_in_objective=SCED_HORIZONS[args.grid],
            lmp_shortfall_costs=False,
            init_ruc_file=ruc_file, 
            verbosity=0,
            output_max_decimals=4, create_plots=False,
            renew_costs=cost_curve, save_to_csv=False, 
            last_conditions_file=None)
    siml.simulate()

    ## rerun simulation with zero reserve shortfall cost
    if args.risk_type != None and args.cost_curve_dir != None:
        init_state_out = Path(args.init_dir, args.grid.lower() + 
            '-init-state-' + (start_date + pd.Timedelta(1, unit='D')
            ).strftime('%Y-%m-%d') + '.csv')
    else:
        init_state_out = None

    siml = Simulator(
            template, gen_data, load_data, None,
            start_date.date(), num_days, solver_options={'Threads': 0}, 
            run_lmps=False, mipgap=RUC_MIPGAPS[args.grid],
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
            renew_costs=cost_curve, save_to_csv=False, 
            last_conditions_file=init_state_out)
    report_dfs = siml.simulate()

    with open(Path(args.out_dir, 'report_dfs-actual.p'), 'wb') as handle:
        pickle.dump(report_dfs, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()

    