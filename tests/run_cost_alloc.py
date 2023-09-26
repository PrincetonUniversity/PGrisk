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
import linecache

from vatic.data.loaders import load_input
from vatic.data.loaders import load_input, RtsLoader, T7kLoader
from pgrisk.cost_attributor import CostAllocator

# Parameters for running VATIC 
RUC_MIPGAPS = {'RTS-GMLC': 0.01, 'Texas-7k': 0.02}
SCED_HORIZONS = {'RTS-GMLC': 4, 'Texas-7k': 2}

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("date", type=str,
                        help="which days to simulate")

    parser.add_argument("grid", type=str, choices={'RTS-GMLC', 'Texas-7k'},
                        help="which power grid to simulate over")

    parser.add_argument("risk_type", type=str, choices={'daily', 'hourly'},
                        help="type of risk, daily or hourly")        

    parser.add_argument("init_dir", 
        type=str, help="initial state directory")
    parser.add_argument("ruc_dir", 
        type=str, help="ruc directory")
    parser.add_argument("reserve_sim_dir", 
        type=str, help="reserve simulation directory")
    parser.add_argument("scen_dir", 
        type=str, help="scenario directory")
    parser.add_argument("cost_alloc_dir", 
        type=str, help="cost allocation directory")

    parser.add_argument("scenario_chunk", type=int, 
                        help="chunk of scenarios for cost allocation")
    parser.add_argument("reserve_factor", type=float, default=0.05, 
                        help="the marginal operating reserve factor used by "
                             "grids in each training simulation")
    parser.add_argument("load_shed_penalty", type=float, default=1e4,
                        help="load shedding penalty per mwh")
    
    args = parser.parse_args()

    # Get scenarios for cost allocation
    try:
        cost_alloc_scenarios = pd.read_pickle(
            Path(args.reserve_sim_dir, args.risk_type + 
            '_cost_alloc_job_chunks.p'))[args.scenario_chunk]
    except:
        cost_alloc_scenarios = []

    for scenario in cost_alloc_scenarios:

        start_date = pd.to_datetime(args.date, utc=True)
        num_days = 1

        ## get baseline data to be day-ahead forecasts
        init_state_in = Path(args.init_dir, 
                args.grid.lower() + '-init-state-' + args.date + '.csv')
        if not init_state_in.exists():
            init_state_in = None

        ## ruc file
        ruc_file = Path(args.ruc_dir, args.grid.lower() + '-ruc-' + args.date + '.p')

        template, baseline_gen_data, baseline_load_data = load_input(args.grid, args.date, 
            num_days=num_days, init_state_file=init_state_in)

        for gen in baseline_gen_data.columns.get_level_values(1).unique():
            baseline_gen_data[('actl', gen)] = baseline_gen_data[('fcst', gen)]

        for load in baseline_load_data.columns.get_level_values(1).unique():
            baseline_load_data[('actl', load)] = baseline_load_data[('fcst', load)]

        ## get scenario data
        if args.grid == 'RTS-GMLC':
            loader = RtsLoader(init_state_file=init_state_in)
        elif args.grid == 'Texas-7k':
            loader = T7kLoader(init_state_file=init_state_in)
        else:
            raise ValueError("unrecognized grid name")

        scen_dfs = loader.load_scenarios(args.scen_dir, [start_date,\
                            start_date + pd.Timedelta(1, unit='D')],
                            [scenario])
        target_gen_data, target_load_data = loader.create_scenario_timeseries(scen_dfs, start_date,
                                            start_date + pd.Timedelta(1, unit='D'), scenario)

        renewable_gens = list(baseline_gen_data.columns.get_level_values(1).unique())
        load_buses = template['Buses']

        tic = time.time()

        if args.grid == 'RTS-GMLC':
            scale = 12
        elif args.grid == 'Texas-7k':
            scale = 8
        else:
            raise ValueError("unrecognized grid name")

        alloc = CostAllocator(
                template_data = template,
                start_date = start_date, 
                mipgap = 0.01,
                reserve_factor = args.reserve_factor, 
                lmp_shortfall_costs = False,
                init_ruc_file = ruc_file, 
                verbosity = 0,
                target_renewables = renewable_gens, 
                target_loads = load_buses,
                baseline_gen_data = baseline_gen_data, 
                baseline_load_data = baseline_load_data, 
                target_gen_data = target_gen_data, 
                target_load_data = target_load_data,
                load_shed_penalty = args.load_shed_penalty,
                reserve_shortfall_penalty = 0,
                solver_options={'Threads': 0},
                scale = scale,
                tol = 0.05,
        )

        alloc.run_adaptive_perturb_analysis()
        alloc.compute_allocation()

        toc = time.time()

        with open(Path(args.cost_alloc_dir, 'summary-' + str(scenario) + '.p'), 'wb') as handle:
            pickle.dump({'time': toc - tic, 'nnods': len(alloc.adaptive_simulations), 
                'summary': alloc.allocation_summary}, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(Path(args.cost_alloc_dir, 'shadow-price-' + str(scenario) + '.p'), 'wb') as handle:
            pickle.dump(alloc.average_shadow_price, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(Path(args.cost_alloc_dir, 'allocation-' + str(scenario) + '.p'), 'wb') as handle:
            pickle.dump(alloc.cost_allocation, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
