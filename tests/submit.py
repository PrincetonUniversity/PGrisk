import os
import argparse
import subprocess
import dill as pickle
import bz2
from pathlib import Path
import random
import pandas as pd
import time

# Parameters for running VATIC 
RUC_MIPGAPS = {'RTS-GMLC': 0.01, 'Texas-7k': 0.02}
SCED_HORIZONS = {'RTS-GMLC': 4, 'Texas-7k': 2}

RUC_TIMES = {'RTS-GMLC': 4, 'Texas-7k': 120}
SCED_TIMES = {'RTS-GMLC': 4, 'Texas-7k': 20}
COST_ALLOC_TIMES = {'RTS-GMLC': 60, 'Texas-7k':1200}
COMPUTE_COST_CURVE_TIMES = {'RTS-GMLC': 10, 'Texas-7k':20}

DELLA_JOB_MIN_TIME = 240

RUC_MEMS = {'RTS-GMLC': '4G', 'Texas-7k': '16G'}
SCED_MEMS = {'RTS-GMLC': '1G', 'Texas-7k': '16G'}
COST_ALLOC_MEMS = {'RTS-GMLC': '8G', 'Texas-7k':'32G'}
COMPUTE_COST_CURVE_MEMS = {'RTS-GMLC': '2G', 'Texas-7k':'4G'}


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("date", type=str,
                        help="which days to simulate")
    
    parser.add_argument("num_days", type=int,
                        help="number of days to simulate")

    parser.add_argument("grid", type=str, choices={'RTS-GMLC', 'Texas-7k'},
                        help="which power grid to simulate over")

    parser.add_argument("sim_dir", type=str, help="simulation directory")

    parser.add_argument("scen_dir", type=str, help="scenario directory")

    parser.add_argument("reserve_factor",
                        type=float, default=0.05,
                        help="the marginal operating reserve factor used by "
                             "grids in each training simulation")
    
    parser.add_argument("nscens",
                        type=int, default=1000,
                        help="number of scenarios to be simulated")

    parser.add_argument("alpha",
                        type=float, default=0.05,
                        help="percentage of scenarios considered to be risky")

    parser.add_argument("risk_type", type=str, choices={'daily', 'hourly'},
                        help="type of risk, daily or hourly")

    parser.add_argument("--load-shed-penalty", dest="load_shed_penalty",
                        type=float, default=1e4,
                        help="load shedding penalty per mwh")

    parser.add_argument("--reserve-shortfall-penalty", dest="reserve_shortfall_penalty",
                        type=float, default=1e3,
                        help="reserve shortfall penalty per mwh")

    parser.add_argument("--num-jobs", dest="num_jobs",
                        type=int, default=100, required=False,
                        help="number of jobs to run reserve scenarios")

    parser.add_argument("--test-scenario", action='store_true', 
                        dest="test_scenario",
                        help="whether to test scenario or not")
    
    parser.add_argument("--remove-existing",
                        action='store_true', dest="remove_existing",
                        help="remove existing intermediate and final output "
                             "folders if present?")

    parser.add_argument("--orfeus", action='store_true', dest="orfeus",
                        help="whether to submit to orfeus or della queue?")

    args = parser.parse_args()

    # create simumation directory
    sim_dir = Path(args.sim_dir)

    if sim_dir.exists() and args.remove_existing:
        subprocess.run(["rm", "-rf", sim_dir])

    subprocess.run(["mkdir", "-p", sim_dir])

    # create log directory
    log_dir = Path(args.sim_dir, "slurm-logs")
    subprocess.run(["mkdir", "-p", log_dir])

    # create init state directory
    init_dir = Path(args.sim_dir, "init-state")
    subprocess.run(["mkdir", "-p", init_dir])

    # create reserve related directories
    reserve_ruc_dir = Path(args.sim_dir, "reserve-ruc")
    subprocess.run(["mkdir", "-p", reserve_ruc_dir])

    reserve_output_dir = Path(args.sim_dir, "reserve-output")
    subprocess.run(["mkdir", "-p", reserve_output_dir])

    # create cost allocation directory
    cost_alloc_dir = Path(args.sim_dir, "cost-alloc")
    subprocess.run(["mkdir", "-p", cost_alloc_dir])

    # create cost curve related directories
    cost_curve_ruc_dir = Path(args.sim_dir, "cost-curve-ruc")
    subprocess.run(["mkdir", "-p", cost_curve_ruc_dir])

    cost_curve_dir = Path(args.sim_dir, "cost-curves")
    subprocess.run(["mkdir", "-p", cost_curve_dir])

    cost_curve_output_dir = Path(args.sim_dir, "cost-curve-output")
    subprocess.run(["mkdir", "-p", cost_curve_output_dir])

    if args.orfeus:
        partition_str = '-p orfeus'
    else:
        partition_str = ''

    for day in pd.date_range(start=args.date, periods=args.num_days, freq='D'):
        
        sim_date = day.strftime('%Y-%m-%d')

        print(sim_date)

        # Check if init_state file exists
        if sim_date != args.date:
            init_state_file = Path(init_dir, args.grid.lower() + 
                '-init-state-' + sim_date + '.csv')
            while not init_state_file.exists():
                time.sleep(10)

        # create directory for sim_date
        sim_log_dir = Path(log_dir, sim_date)
        subprocess.run(["mkdir", "-p", sim_log_dir])

        sim_reserve_output_dir = Path(reserve_output_dir, sim_date)
        subprocess.run(["mkdir", "-p", sim_reserve_output_dir])

        sim_cost_alloc_dir = Path(cost_alloc_dir, sim_date)
        subprocess.run(["mkdir", "-p", sim_cost_alloc_dir])

        sim_cost_curve_output_dir = Path(cost_curve_output_dir, sim_date)
        subprocess.run(["mkdir", "-p", sim_cost_curve_output_dir])

        ## Run day-ahead ruc
        JOB_TIME = max(DELLA_JOB_MIN_TIME,
            RUC_TIMES[args.grid] + SCED_TIMES[args.grid])
        init_ruc_job = subprocess.run(
            f'sbatch --parsable --job-name={"run_init_ruc-" + sim_date} '
            f'--nodes=1 --ntasks=1 --cpus-per-task=4 '
            f'--time={JOB_TIME} '
            f'--mem-per-cpu={RUC_MEMS[args.grid]} '
            f'{partition_str} '
            f'--wrap=" python run_init_ruc.py {sim_date} {args.grid} '
            f'{init_dir} {reserve_ruc_dir} {sim_reserve_output_dir} {args.reserve_factor} '
            f'{args.load_shed_penalty} {args.reserve_shortfall_penalty}" '
            f'--output={Path(sim_log_dir, "init-ruc-%A.out")} '
            f'--error={Path(sim_log_dir, "init-ruc-%A.err")} ',

            shell=True, capture_output=True
            ).stdout.decode('utf-8').strip()

        dep_str = (f"--dependency=afterok:{init_ruc_job} "
                    "--kill-on-invalid-dep=yes")

        JOB_TIME = max(DELLA_JOB_MIN_TIME,
            int(args.nscens // args.num_jobs) * SCED_TIMES[args.grid])
        run_reserve_scen_job = subprocess.run(
                f'sbatch --parsable --job-name={"run_reserve_scen-" + sim_date} '
                f'--array=0-{args.num_jobs - 1} '
                f'--time={JOB_TIME} '
                f'--nodes=1 --ntasks=1 --cpus-per-task=1 '
                f'--mem-per-cpu={SCED_MEMS[args.grid]} {dep_str} '
                f'{partition_str} '
                f'--wrap=" python run_scenario.py {sim_date} {args.grid} '
                f'{args.scen_dir} {init_dir} {reserve_ruc_dir} '
                f'{sim_reserve_output_dir} {args.nscens} {args.num_jobs} \$SLURM_ARRAY_TASK_ID '
                f'{args.reserve_factor} {args.load_shed_penalty}" '
                f'--output={Path(sim_log_dir, "reserve-scen-%A-%a.out")} '
                f'--error={Path(sim_log_dir, "reserve-scen-%A-%a.err")} ',

                shell=True, capture_output=True
                ).stdout.decode('utf-8').strip()

        dep_str = (f"--dependency=afterok:{run_reserve_scen_job} "
                    "--kill-on-invalid-dep=yes")

        ## Get risky scenarios
        JOB_TIME = max(DELLA_JOB_MIN_TIME,
            RUC_TIMES[args.grid] + SCED_TIMES[args.grid])
        get_risky_scens_job = subprocess.run(
            f'sbatch --parsable --job-name={"get_risky_scens-" + sim_date} '
            f'--nodes=1 --ntasks=1 --cpus-per-task=1 '
            f'--time={JOB_TIME} '
            f'--mem-per-cpu={RUC_MEMS[args.grid]} {dep_str} '
            f'{partition_str} '
            f'--wrap=" python get_risky_scens.py {args.nscens} '
            f'{args.alpha} {args.risk_type} '
            f'{sim_reserve_output_dir} {args.num_jobs} '
            f'{args.load_shed_penalty} {args.reserve_shortfall_penalty}" '
            f'--output={Path(sim_log_dir, "get-risky-scens-%A.out")} '
            f'--error={Path(sim_log_dir, "get-risky-scens-%A.err")} ',

            shell=True, capture_output=True
            ).stdout.decode('utf-8').strip()

        dep_str = (f"--dependency=afterok:{get_risky_scens_job} "
                    "--kill-on-invalid-dep=yes")

        ## Submit cost allocation jobs
        JOB_TIME = max(DELLA_JOB_MIN_TIME, COST_ALLOC_TIMES[args.grid])
        cost_alloc_job = subprocess.run(
            f'sbatch --parsable --job-name={"run_cost_alloc-" + sim_date} '
            f'--array=0-{args.num_jobs - 1} '
            f'--nodes=1 --cpus-per-task=1 '
            f'--time={JOB_TIME} '
            f'--mem-per-cpu={COST_ALLOC_MEMS[args.grid]} {dep_str} '
            f'{partition_str} '
            f'--wrap=" python run_cost_alloc.py {sim_date} '
            f'{args.grid} {args.risk_type} {init_dir} '
            f'{reserve_ruc_dir} {sim_reserve_output_dir} {args.scen_dir} '
            f'{sim_cost_alloc_dir} '
            f'\$SLURM_ARRAY_TASK_ID {args.reserve_factor} {args.load_shed_penalty}" '
            f'--output={Path(sim_log_dir, "cost-alloc-%A-%a.out")} '
            f'--error={Path(sim_log_dir, "cost-alloc-%A-%a.err")} ',

            shell=True, capture_output=True
            ).stdout.decode('utf-8').strip()

        dep_str = (f"--dependency=afterok:{cost_alloc_job} "
                    "--kill-on-invalid-dep=yes")

        ## Compute cost curves
        JOB_TIME = max(DELLA_JOB_MIN_TIME, COMPUTE_COST_CURVE_TIMES[args.grid])
        compute_cost_curve_job = subprocess.run(
            f'sbatch --parsable --job-name={"compute_cost_curve-" + sim_date} '
            f'--nodes=1 --cpus-per-task=1 '
            f'--time={JOB_TIME} '
            f'--mem-per-cpu={COMPUTE_COST_CURVE_MEMS[args.grid]} {dep_str} '
            f'{partition_str} '
            f'--wrap=" python compute_cost_curve.py {sim_date} '
            f'{args.grid} {args.risk_type} {args.nscens} '
            f'{args.alpha} {args.scen_dir} {sim_cost_alloc_dir} '
            f'{sim_reserve_output_dir} {cost_curve_dir}" '
            f'--output={Path(sim_log_dir, "compute-cost-curve-%A.out")} '
            f'--error={Path(sim_log_dir, "compute-cost-curve-%A.err")} ',

            shell=True, capture_output=True
            ).stdout.decode('utf-8').strip()

        dep_str = (f"--dependency=afterok:{compute_cost_curve_job} "
                    "--kill-on-invalid-dep=yes")

        ## Run ruc with cost curve
        JOB_TIME = max(DELLA_JOB_MIN_TIME, RUC_TIMES[args.grid] + SCED_TIMES[args.grid])
        cost_curve_init_ruc_job = subprocess.run(
            f'sbatch --parsable --job-name={"run_cost_curve_init_ruc-" + sim_date} '
            f'--nodes=1 --ntasks=1 --cpus-per-task=4 '
            f'--time={JOB_TIME} '
            f'--mem-per-cpu={RUC_MEMS[args.grid]} {dep_str} '
            f'{partition_str} '
            f'--wrap=" python run_init_ruc.py {sim_date} {args.grid} '
            f'{init_dir} {cost_curve_ruc_dir} {sim_cost_curve_output_dir} '
            f'{args.reserve_factor} {args.load_shed_penalty} {args.reserve_shortfall_penalty} '
            f'--risk-type {args.risk_type} --cost-curve-dir {cost_curve_dir} " '
            f'--output={Path(sim_log_dir, "cost-curve-init-ruc-%A.out")} '
            f'--error={Path(sim_log_dir, "cost-curve-init-ruc-%A.err")} ',

            shell=True, capture_output=True
            ).stdout.decode('utf-8').strip()

        dep_str = (f"--dependency=afterok:{cost_curve_init_ruc_job} "
                    "--kill-on-invalid-dep=yes")

        if args.test_scenario:
            JOB_TIME = max(DELLA_JOB_MIN_TIME, 
                int(args.nscens // args.num_jobs) * SCED_TIMES[args.grid])
            run_cost_curve_scen_job = subprocess.run(
                f'sbatch --parsable --job-name={"run_cost_curve_scen-" + sim_date} '
                f'--array=0-{args.num_jobs - 1} '
                f'--time={JOB_TIME} '
                f'--nodes=1 --ntasks=1 --cpus-per-task=1 '
                f'--mem-per-cpu={SCED_MEMS[args.grid]} {dep_str} '
                f'{partition_str} '
                f'--wrap=" python run_scenario.py {sim_date} {args.grid} '
                f'{args.scen_dir} {init_dir} {cost_curve_ruc_dir} '
                f'{sim_cost_curve_output_dir} {args.nscens} {args.num_jobs} \$SLURM_ARRAY_TASK_ID '
                f'{args.reserve_factor} {args.load_shed_penalty}" '
                f'--output={Path(sim_log_dir, "cost-curve-scen-%A-%a.out")} '
                f'--error={Path(sim_log_dir, "cost-curve-scen-%A-%a.err")} ',

                shell=True, capture_output=True
                ).stdout.decode('utf-8').strip()

            dep_str = (f"--dependency=afterok:{run_cost_curve_scen_job} "
                        "--kill-on-invalid-dep=yes")
                        

if __name__ == '__main__':
    main()