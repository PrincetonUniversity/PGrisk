import os
import argparse
import subprocess
import dill as pickle
import bz2
from pathlib import Path
import random
import pandas as pd
from vatic.data.loaders import load_input
from vatic.engines import Simulator

# Parameters for running VATIC 
RUC_MIPGAPS = {'RTS-GMLC': 0.01, 'Texas7k': 0.02}
SCED_HORIZONS = {'RTS-GMLC': 4, 'Texas7k': 2}

def main():

    start_date = '2020-01-02'
    num_days = 364
    grid = 'RTS-GMLC'
    sim_dir = '/projects/PERFORM/xyang/data/cost_alloc/RTS-GMLC/reserve-30/'
    reserve_factor = 0.3
    remove_existing = True

    # create simumation directory
    sim_dir = Path(sim_dir)

    if sim_dir.exists() and remove_existing:
        subprocess.run(["rm", "-rf", sim_dir])

    subprocess.run(["mkdir", "-p", sim_dir])

    # create init state directory
    init_dir = Path(sim_dir, "init-state")
    subprocess.run(["mkdir", "-p", init_dir])

    # create reserve related directories
    reserve_ruc_dir = Path(sim_dir, "reserve-ruc")
    subprocess.run(["mkdir", "-p", reserve_ruc_dir])

    reserve_output_dir = Path(sim_dir, 
        "reserve-output")
    subprocess.run(["mkdir", "-p", reserve_output_dir])

    for day in pd.date_range(start=start_date, 
            periods=num_days, freq='D', tz='utc'):

        if day.strftime('%Y-%m-%d') == '2020-01-02':
            init_state_in = None
        else:
            init_state_in = Path(init_dir, grid.lower() + 
                '-init-state-' + day.strftime('%Y-%m-%d') + '.csv')

        template, gen_data, load_data = load_input(grid, 
            day.strftime('%Y-%m-%d'), num_days=1, 
            init_state_file=init_state_in)

        siml = Simulator(
                template, gen_data,load_data, None, day.date(), 1, 
                solver_options={'Threads': 0}, run_lmps=False, mipgap=RUC_MIPGAPS[grid],
                load_shed_penalty = 1e4, 
                reserve_shortfall_penalty = 1e3,
                reserve_factor=reserve_factor, output_detail=3,
                prescient_sced_forecasts=True, ruc_prescience_hour=0,
                ruc_execution_hour=16, ruc_every_hours=24,
                ruc_horizon=48, sced_horizon=SCED_HORIZONS[grid],
                hours_in_objective=SCED_HORIZONS[grid],
                lmp_shortfall_costs=False,
                init_ruc_file=Path(reserve_ruc_dir, 
                    grid.lower() + '-init-ruc-' + 
                    day.strftime('%Y-%m-%d') + '.p'), 
                verbosity=0,
                output_max_decimals=4, create_plots=False,
                renew_costs=None, save_to_csv=False, 
                last_conditions_file=None
        )
        siml.simulate()

        ## rerun simulation with zero reserve shortfall cost
        siml = Simulator(
                template, gen_data, load_data, None, day.date(), 1, 
                solver_options={'Threads': 0}, run_lmps=True, mipgap=RUC_MIPGAPS[grid],
                load_shed_penalty = 1e4, 
                reserve_shortfall_penalty = 0,
                reserve_factor=reserve_factor, output_detail=3,
                prescient_sced_forecasts=True, ruc_prescience_hour=0,
                ruc_execution_hour=16, ruc_every_hours=24,
                ruc_horizon=48, sced_horizon=SCED_HORIZONS[grid],
                hours_in_objective=SCED_HORIZONS[grid],
                lmp_shortfall_costs=False,
                init_ruc_file=Path(reserve_ruc_dir, 
                    grid.lower() + '-init-ruc-' + 
                    day.strftime('%Y-%m-%d') + '.p'),  
                verbosity=0,
                output_max_decimals=4, create_plots=False,
                renew_costs=None, save_to_csv=False, 
                last_conditions_file=Path(init_dir, 
                        grid.lower() + '-init-state-' + 
                        (day + pd.Timedelta(1, unit='D')).strftime('%Y-%m-%d') + '.csv')
        )
        report_dfs = siml.simulate()

        with open(Path(reserve_output_dir, 
                'report_dfs-' + day.strftime('%Y-%m-%d') + 
                '.p'), 'wb') as handle:
            pickle.dump(report_dfs, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        

if __name__ == '__main__':
    main()