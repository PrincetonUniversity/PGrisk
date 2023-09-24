from __future__ import annotations

from typing import Union
from pathlib import Path
import pandas as pd
import numpy as np
import math
import os
import time
from datetime import datetime, timedelta
import shutil
from copy import deepcopy
from functools import reduce
from .shadow_price import ShadowPriceSimulator


class CostAttributor:
    """This abstract class attributes cost difference of running 
    SCEDs between the baseline and the target inputs.
    """

    def __init__(self, template_data: dict, start_date: datetime.date, 
                mipgap: float, reserve_factor: float, lmp_shortfall_costs: bool,
                init_ruc_file: str | Path | None, verbosity: int, 
                renewable_gens: list[str], load_buses: list[str],
                baseline_gen_data: pd.DataFrame, baseline_load_data: pd.DataFrame, 
                target_gen_data: pd.DataFrame, target_load_data: pd.DataFrame,
                load_shed_penalty: float, reserve_shortfall_penalty: float,
                workdir: Path | str, scale :int = 8, tol : float = 0.05,
                ) -> None:

        self.template_data = template_data
        self.start_date = start_date
        self.mipgap = mipgap
        self.reserve_factor = reserve_factor
        self.lmp_shortfall_costs = lmp_shortfall_costs
        self.init_ruc_file = Path(init_ruc_file)
        self.verbosity = verbosity

        self.renewable_gens = renewable_gens
        self.load_buses = load_buses
        self.baseline_gen_data = baseline_gen_data
        self.baseline_load_data = baseline_load_data
        self.target_gen_data = target_gen_data
        self.target_load_data = target_load_data
        self.workdir = Path(workdir)
        self.load_shed_penalty = load_shed_penalty
        self.reserve_shortfall_penalty = reserve_shortfall_penalty
        self.scale = scale
        self.tol = tol
        self.adaptive_simulations = {}

        self.baseline_report_dfs = None
        self.target_report_dfs = None
        self.ruc_every_hours = 24

        ## get ThermalGeneratorsAtBus
        self.ThermalGeneratorsAtBus = dict()
        for bus in self.template_data['ThermalGeneratorsAtBus']:
            if self.template_data['ThermalGeneratorsAtBus'][bus]:
                for gen in self.template_data['ThermalGeneratorsAtBus'][bus]:
                    self.ThermalGeneratorsAtBus[gen] = bus
        
        ## compute number of nodes and step size
        self.nnods = 2 ** self.scale
        self.stepsize = 1. / self.nnods
        self.simulations = {
            nod : {'alpha' : nod * self.stepsize, 
                   'complete': False, 
                   'shadow_price': None} 
            for nod in range(self.nnods + 1)
        }

        ## time step for computing shadow price
        self.shadow_price_timesteps = [
            self.start_date.to_pydatetime().replace(tzinfo=None) + timedelta(hours=hour) 
            for hour in range(self.ruc_every_hours)]

        ## get gen and load difference
        self.gen_diff = self.target_gen_data - self.baseline_gen_data
        self.load_diff = self.target_load_data - self.baseline_load_data

        act_gen_diff = self.gen_diff.loc[pd.to_datetime(self.shadow_price_timesteps, utc=True), 
                                [('actl', asset) for asset in self.renewable_gens]].copy(deep=True)
        act_gen_diff.fillna(method='ffill', inplace=True)
        act_gen_diff.columns = act_gen_diff.columns.droplevel(0)
        act_gen_diff.index = act_gen_diff.index.map(lambda x: x.to_pydatetime().replace(tzinfo=None))

        act_load_diff = self.load_diff.loc[pd.to_datetime(self.shadow_price_timesteps, utc=True), 
                                    [('actl', load) for load in self.load_buses]].copy(deep=True)
        act_load_diff.fillna(method='ffill', inplace=True)
        act_load_diff.columns = act_load_diff.columns.droplevel(0)
        act_load_diff.index = act_load_diff.index.map(lambda x: x.to_pydatetime().replace(tzinfo=None))

        self.act_gen_diff = act_gen_diff
        self.act_load_diff = act_load_diff

    def clean_wkdir(self):
        shutil.rmtree(self.workdir)

    def simulate(self, nod: int, clean_wkdir: bool = True) -> dict:
        """Run simulation at given node and compute shadow price"""

        simulator = ShadowPriceSimulator(self.load_buses, self.renewable_gens,
                Path(self.workdir, str(nod)), template_data = self.template_data, 
                gen_data = self.baseline_gen_data + self.simulations[nod]['alpha'] * self.gen_diff, 
                load_data = self.baseline_load_data + self.simulations[nod]['alpha'] * self.load_diff,
                out_dir = None, start_date = self.start_date.date(), num_days = 1, 
                solver = 'gurobi', solver_options = None, run_lmps = False, mipgap = self.mipgap, 
                load_shed_penalty = self.load_shed_penalty, 
                reserve_shortfall_penalty = self.reserve_shortfall_penalty,
                reserve_factor = self.reserve_factor, output_detail = 2, 
                prescient_sced_forecasts = True, ruc_prescience_hour = 0,
                ruc_execution_hour = 16, ruc_every_hours = 24, ruc_horizon = 48,
                sced_horizon = 1, lmp_shortfall_costs = False, enforce_sced_shutdown_ramprate = False,
                no_startup_shutdown_curves = False, init_ruc_file = self.init_ruc_file, 
                verbosity = 0, output_max_decimals = 4, create_plots = False, 
                renew_costs = None, save_to_csv = False, last_conditions_file = None,)

        simulator.simulate()
        simulator.simulate_shadow_price()

        cost_df = deepcopy(simulator.cost_df)
        shadow_price = deepcopy(simulator.shadow_price)
        init_thermal_production = deepcopy(simulator.init_thermal_production)
        unpriced_thermal_production = simulator.unpriced_thermal_production

        if nod == 0:
            self.baseline_report_dfs = simulator.report_dfs
        if nod == self.nnods:
            self.target_report_dfs = simulator.report_dfs

        # clean up
        if clean_wkdir:
            simulator.clean_wkdir()
        del simulator

        return {'nod' : nod, 'costs' : cost_df, 'shadow_price' : shadow_price, 
            'init_thermal_production' : init_thermal_production,
            'unpriced_thermal_production': unpriced_thermal_production}

    def run_shadow_price_analysis(self, left : int = None, 
            right : int = None, clean_wkdir : bool = True) -> None:
    
        if left is None:
            left = 0
        if right is None:
            right = self.nnods

        average_shadow_price = {}
        cost_attribution = {}
        
        if not self.simulations[left]['complete']:
            results = self.simulate(left, clean_wkdir)
            self.simulations[left]['costs'] = results['costs']
            self.simulations[left]['shadow_price'] = results['shadow_price']
            self.simulations[left]['init_thermal_production'] = results['init_thermal_production']
            self.simulations[left]['complete'] = True
            self.simulations[left]['unpriced_thermal_production'] = results['unpriced_thermal_production']
            
        if not self.simulations[right]['complete']:
            results = self.simulate(right, clean_wkdir)
            self.simulations[right]['costs'] = results['costs']
            self.simulations[right]['shadow_price'] = results['shadow_price']
            self.simulations[right]['init_thermal_production'] = results['init_thermal_production']
            self.simulations[right]['complete'] = True
            self.simulations[right]['unpriced_thermal_production'] = results['unpriced_thermal_production']
            
        ## compute average shadow price
        for gen in self.renewable_gens:
            arr = []
            for timestep in self.shadow_price_timesteps:
                shadow_price = 0.5 * (
                                self.simulations[left]['shadow_price'][timestep][gen] + \
                                self.simulations[right]['shadow_price'][timestep][gen])
                arr.append(shadow_price)
                
            average_shadow_price[gen] = pd.DataFrame({'average_shadow_price': arr},
                            index = self.shadow_price_timesteps)

        for load in self.load_buses:
            arr = []
            for timestep in self.shadow_price_timesteps:
                shadow_price = 0.5 * (
                                self.simulations[left]['shadow_price'][timestep][load] + \
                                self.simulations[right]['shadow_price'][timestep][load])
                arr.append(shadow_price)
            average_shadow_price[load] = pd.DataFrame({'average_shadow_price': arr},
                            index = self.shadow_price_timesteps)
        
        ## compute attribution for renewable gens     
        shadow_price_adjustment = (right - left) / self.nnods

        allocated_gen_costs = pd.DataFrame(
            {'costs': np.zeros(self.ruc_every_hours)}, 
            index=self.shadow_price_timesteps
        )

        for gen in self.renewable_gens:
            cost_attribution[gen] = average_shadow_price[gen].copy(
                deep=True).rename(columns = {'average_shadow_price' : 'costs'})

            for timestep in cost_attribution[gen].index:
                cost_attribution[gen].loc[timestep] *= shadow_price_adjustment * \
                    self.act_gen_diff.loc[timestep, gen]

            allocated_gen_costs += cost_attribution[gen]

        ## compute attribution for loads
        allocated_load_costs = pd.DataFrame(
            {'costs': np.zeros(self.ruc_every_hours)}, 
            index=self.shadow_price_timesteps
        )
        for load in self.load_buses:
            cost_attribution[load] = average_shadow_price[load].copy(
                deep=True).rename(columns = {'average_shadow_price' : 'costs'})

            for timestep in cost_attribution[load].index:
                cost_attribution[load].loc[timestep] *= shadow_price_adjustment * \
                    self.act_load_diff.loc[timestep, load]

            allocated_load_costs += cost_attribution[load]

        ## compute attribution for untracked thermal gen
        allocated_thermal_costs = pd.DataFrame(
            {'costs': np.zeros(self.ruc_every_hours)}, 
            index=self.shadow_price_timesteps
        )
        for gen in self.template_data['ThermalGenerators']:
            load = self.ThermalGeneratorsAtBus[gen]
            cost_attribution[gen] = - average_shadow_price[load].copy(
                deep=True).rename(columns = {'average_shadow_price' : 'costs'})

            for timestep in self.shadow_price_timesteps:
                cost_attribution[gen].loc[timestep] *= (
                    self.simulations[right]['unpriced_thermal_production'][gen].get(timestep, 0.) - \
                    self.simulations[left]['unpriced_thermal_production'][gen].get(timestep, 0.)
                )

            allocated_thermal_costs += cost_attribution[gen]

        ## compute initial condition costs attribution
        init_costs = []
        for timestep in self.shadow_price_timesteps:
            costs = 0.
            for gen in self.template_data['ThermalGenerators']:
                ## up
                shadow_price = 0.5 * (self.simulations[left]['shadow_price'][timestep].get(gen + '_Up', 0.) + \
                        self.simulations[right]['shadow_price'][timestep].get(gen + '_Up', 0.))
                costs += shadow_price * (self.simulations[right]['init_thermal_production'][timestep][gen] - \
                            self.simulations[left]['init_thermal_production'][timestep][gen])
                
                ## down
                shadow_price = 0.5 * (self.simulations[left]['shadow_price'][timestep].get(gen + '_Down', 0.) + \
                        self.simulations[right]['shadow_price'][timestep].get(gen + '_Down', 0.))
                costs += shadow_price * (self.simulations[right]['init_thermal_production'][timestep][gen] - \
                            self.simulations[left]['init_thermal_production'][timestep][gen])
                    
            init_costs.append(costs)

        attribution_summary = self.simulations[right]['costs'] - self.simulations[left]['costs']
        attribution_summary['renewable_costs'] = allocated_gen_costs['costs'].values
        attribution_summary['load_costs'] = allocated_load_costs['costs'].values
        attribution_summary['unpriced_thermal_costs'] = allocated_thermal_costs['costs'].values
        attribution_summary['initial_thernal_costs'] = init_costs
        attribution_summary['attribution'] = attribution_summary['renewable_costs'] + \
                attribution_summary['load_costs'] + attribution_summary['initial_thernal_costs'] + \
                attribution_summary['unpriced_thermal_costs']
        attribution_summary = attribution_summary[['total_costs', 'attribution',
                    'renewable_costs', 'load_costs', 'unpriced_thermal_costs', 
                    'initial_thernal_costs']]
        attribution_summary.index.name = 'Time'

        ## compute absolute error threshold
        if not hasattr(self, 'abs_tol'):
            df = self.simulations[self.nnods]['costs'] - self.simulations[0]['costs']
            self.abs_tol = self.tol * df['total_costs'].abs().max()

        ## compute attribution error
        err = (attribution_summary['total_costs'] - attribution_summary['attribution']).abs().max()
        
        if ( err <= (right - left) / self.nnods * self.abs_tol ) or ( right - left == 1 ):
            self.adaptive_simulations[(left, right)] = {'average_shadow_price' : average_shadow_price,
                                                        'cost_attribution' : cost_attribution,
                                                        'attribution_summary' : attribution_summary}
        else:
            ## recursion
            mid = left + (right - left) // 2
            self.run_shadow_price_analysis(left, mid, clean_wkdir)
            self.run_shadow_price_analysis(mid, right, clean_wkdir)


    def compute_attribution(self):
        ## make sure nodes covers entire grids
        adaptive_grid = sorted(self.adaptive_simulations.keys(), key = lambda x : x[1])
        if adaptive_grid[0][0] != 0 or adaptive_grid[-1][-1] != self.nnods:
            raise RuntimeError('adaptive grid do not completely cover entire grid')
        for i in range(len(adaptive_grid) - 1):
            if adaptive_grid[i][-1] != adaptive_grid[i + 1][0]:
                raise RuntimeError('adaptive grid do not completely cover entire grid')

        average_shadow_price = {}
        cost_attribution = {}
        for asset in self.renewable_gens + self.load_buses:
            average_shadow_price[asset] = reduce(lambda x, y: x.add(y, fill_value=0), 
                [self.adaptive_simulations[left, right]['average_shadow_price'][asset] * \
                    (right - left) / self.nnods for (left, right) in adaptive_grid])
            cost_attribution[asset] = reduce(lambda x, y: x.add(y, fill_value=0), 
                [self.adaptive_simulations[nod]['cost_attribution'][asset] for nod in adaptive_grid])

        self.average_shadow_price = average_shadow_price
        self.cost_attribution = cost_attribution
        self.attribution_summary = reduce(lambda x, y: x.add(y, fill_value=0), 
            [self.adaptive_simulations[nod]['attribution_summary'] for nod in adaptive_grid]).round(2)
