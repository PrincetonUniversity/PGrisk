from __future__ import annotations

from typing import Union
from pathlib import Path
import pandas as pd
import numpy as np
import math
import os
import time
from datetime import datetime, timedelta
import dill as pickle
import shutil
from copy import deepcopy
import multiprocessing as mp
from functools import reduce

import gurobipy as gp
from gurobipy import GRB

from .shadow_price import ShadowPriceSimulator


class CostAllocator:
    """This abstract class allocates cost of running SCED 
    between the baseline and the target scenarios using perturbation analysis.
    """

    def __init__(self, 
                template_data: dict,
                start_date: datetime.date, 
                mipgap: float,
                reserve_factor: float, 
                lmp_shortfall_costs: bool,
                init_ruc_file: str | Path | None, 
                verbosity: int,
                target_assets: list[str], 
                target_loads: list[str],
                baseline_gen_data: pd.DataFrame, 
                baseline_load_data: pd.DataFrame, 
                target_gen_data: pd.DataFrame, 
                target_load_data: pd.DataFrame,
                load_shed_penalty: float,
                reserve_shortfall_penalty: float,
                scale :int = 8,
                tol : float = 0.05
                ) -> None:

        self.template_data = template_data
        self.start_date = start_date
        self.mipgap = mipgap
        self.reserve_factor = reserve_factor
        self.lmp_shortfall_costs = lmp_shortfall_costs
        self.init_ruc_file = Path(init_ruc_file)
        self.verbosity = verbosity

        self.target_assets = target_assets
        self.target_loads = target_loads
        self.baseline_gen_data = baseline_gen_data
        self.baseline_load_data = baseline_load_data
        self.target_gen_data = target_gen_data
        self.target_load_data = target_load_data
        self.load_shed_penalty = load_shed_penalty
        self.reserve_shortfall_penalty = reserve_shortfall_penalty
        self.scale = scale
        self.rel_tol = tol
        self.abs_tol = None
        self.adaptive_simulations = {}

        self.baseline_report_dfs = None
        self.target_report_dfs = None

        ## hard-coded parameters
        self.ruc_every_hours = 24

        ## get ThermalGeneratorsAtBus
        self.ThermalGeneratorsAtBus = dict()
        for bus in self.template_data['ThermalGeneratorsAtBus']:
            if self.template_data['ThermalGeneratorsAtBus'][bus]:
                for gen in self.template_data['ThermalGeneratorsAtBus'][bus]:
                    self.ThermalGeneratorsAtBus[gen] = bus
        
        ## 
        self.nnods = 2 ** self.scale
        self.stepsize = 1. / self.nnods
        self.simulations = {
            nod : {'alpha' : nod * self.stepsize, 
                    'simulated': False, 
                    'shadow_price': None
            } 
            for nod in range(self.nnods + 1)
        }

        ## time step when renewable production to be perturbed 
        self.perturb_timesteps = [
            (self.start_date.to_pydatetime().replace(tzinfo=None) + timedelta(hours=hour)) 
            for hour in range(self.ruc_every_hours)]

        ## time step when perturbations affect costs
        self.effect_timesteps = [
            self.start_date.to_pydatetime().replace(tzinfo=None) + timedelta(hours=hour) 
            for hour in range(self.ruc_every_hours)]

        ## get gen and load difference
        self.gen_diff = self.target_gen_data - self.baseline_gen_data
        self.load_diff = self.target_load_data - self.baseline_load_data

        act_gen_diff = self.gen_diff.loc[pd.to_datetime(self.perturb_timesteps, utc=True), 
                                [('actl', asset) for asset in self.target_assets]].copy(deep=True)
        act_gen_diff.fillna(method='ffill', inplace=True)
        act_gen_diff.columns = act_gen_diff.columns.droplevel(0)
        act_gen_diff.index = act_gen_diff.index.map(lambda x: x.to_pydatetime().replace(tzinfo=None))

        act_load_diff = self.load_diff.loc[pd.to_datetime(self.perturb_timesteps, utc=True), 
                                    [('actl', load) for load in self.target_loads]].copy(deep=True)
        act_load_diff.fillna(method='ffill', inplace=True)
        act_load_diff.columns = act_load_diff.columns.droplevel(0)
        act_load_diff.index = act_load_diff.index.map(lambda x: x.to_pydatetime().replace(tzinfo=None))

        self.act_gen_diff = act_gen_diff
        self.act_load_diff = act_load_diff

    def simulate(self, nod: int) -> dict:
        """Run simulation at given node and compute shadow price"""

        simulator = ShadowPriceSimulator(
                load_buses = self.target_loads,
                renewable_gens = self.target_assets,
                template_data = self.template_data, 
                gen_data = self.baseline_gen_data + self.simulations[nod]['alpha'] * self.gen_diff, 
                load_data = self.baseline_load_data + self.simulations[nod]['alpha'] * self.load_diff,
                out_dir = None, start_date = self.start_date.date(), num_days = 1, 
                solver_options = {'Threads':8}, run_lmps = False, mipgap = 1e-4, 
                load_shed_penalty = self.load_shed_penalty, 
                reserve_shortfall_penalty = self.reserve_shortfall_penalty,
                reserve_factor = self.reserve_factor,
                output_detail = 3,
                prescient_sced_forecasts = True, 
                ruc_prescience_hour = 0,
                ruc_execution_hour = 16, 
                ruc_every_hours = 24, 
                ruc_horizon = 48,
                sced_horizon = 1, 
                lmp_shortfall_costs = False,
                init_ruc_file = self.init_ruc_file, 
                verbosity = 0,
                output_max_decimals = 4, 
                create_plots = False, 
                renew_costs = None,
                save_to_csv = False,
                last_conditions_file = None,
                )

        report_dfs = simulator.simulate()
        print(report_dfs['hourly_summary'].iloc[1, 1])

        cost_df = simulator.cost_df.copy(deep=True)
        shadow_price = deepcopy(simulator.shadow_price)
        init_thermal_output = deepcopy(simulator.init_thermal_output)
        unpriced_thermal_production = simulator.unpriced_thermal_production

        if nod == 0:
            self.baseline_report_dfs = report_dfs
        elif nod == self.nnods:
            self.target_report_dfs = report_dfs

        del simulator

        return {'nod' : nod, 'costs' : cost_df, 'shadow_price' : shadow_price, 
            'init_thermal_output' : init_thermal_output,
            'unpriced_thermal_production': unpriced_thermal_production}

    def run_perturb_analysis(self, 
                             processes: int | None = None, 
                             clean_wkdir : bool = True) -> None:

        if not processes:
            for nod in range(self.nnods + 1):
                print(nod)
                simulation_results = self.simulate(nod, clean_wkdir)

                self.simulations[nod]['costs'] = simulation_results['costs']
                self.simulations[nod]['shadow_price'] = simulation_results['shadow_price']
                self.simulations[nod]['init_thermal_output'] = simulation_results['init_thermal_output']
                self.simulations[nod]['simulated'] = True
        else:
            if processes > mp.cpu_count():
                processes = mp.cpu_count()

            with mp.Pool(processes=processes) as pool:
                for result in pool.starmap(
                        self.simulate, 
                        [(nod, clean_wkdir) for nod in range(self.nnods + 1)]):

                    nod = result['nod']
                    self.simulations[nod]['costs'] = result['costs']
                    self.simulations[nod]['shadow_price'] = result['shadow_price']
                    self.simulations[nod]['init_thermal_output'] = result['init_thermal_output']
                    self.simulations[nod]['simulated'] = True



    def run_adaptive_perturb_analysis(
            self, left : int = None, right : int = None) -> None:
    
        if left is None:
            left = 0
        if right is None:
            right = self.nnods

        average_shadow_price = {}
        cost_allocation = {}
        
        if not self.simulations[left]['simulated']:
            results = self.simulate(left)
            self.simulations[left]['costs'] = results['costs']
            self.simulations[left]['shadow_price'] = results['shadow_price']
            self.simulations[left]['init_thermal_output'] = results['init_thermal_output']
            self.simulations[left]['simulated'] = True
            self.simulations[left]['unpriced_thermal_production'] = results['unpriced_thermal_production']
            
        if not self.simulations[right]['simulated']:
            results = self.simulate(right)
            self.simulations[right]['costs'] = results['costs']
            self.simulations[right]['shadow_price'] = results['shadow_price']
            self.simulations[right]['init_thermal_output'] = results['init_thermal_output']
            self.simulations[right]['simulated'] = True
            self.simulations[right]['unpriced_thermal_production'] = results['unpriced_thermal_production']
            
        ## compute average shadow price
        for gen in self.target_assets:
            arr = []
            for timestep in self.perturb_timesteps:
                shadow_price = 0.5 * (
                                self.simulations[left]['shadow_price'][timestep][gen] + \
                                self.simulations[right]['shadow_price'][timestep][gen])
                arr.append(shadow_price)
                
            average_shadow_price[gen] = pd.DataFrame({'average_shadow_price': arr},
                            index = self.effect_timesteps)

        for load in self.target_loads:
            arr = []
            for timestep in self.perturb_timesteps:
                shadow_price = 0.5 * (
                                self.simulations[left]['shadow_price'][timestep][load] + \
                                self.simulations[right]['shadow_price'][timestep][load])
                arr.append(shadow_price)
            average_shadow_price[load] = pd.DataFrame({'average_shadow_price': arr},
                            index = self.effect_timesteps)
        
        ## compute allocation for renewable gens     
        shadow_price_adjustment = (right - left) / self.nnods

        allocated_gen_costs = pd.DataFrame(
            {'costs': np.zeros(self.ruc_every_hours)}, 
            index=self.perturb_timesteps
        )

        for gen in self.target_assets:
            cost_allocation[gen] = average_shadow_price[gen].copy(
                deep=True).rename(columns = {'average_shadow_price' : 'costs'})

            for timestep in cost_allocation[gen].index:
                cost_allocation[gen].loc[timestep] *= shadow_price_adjustment * \
                    self.act_gen_diff.loc[timestep, gen]

            allocated_gen_costs += cost_allocation[gen]

        ## compute allocation for loads
        allocated_load_costs = pd.DataFrame(
            {'costs': np.zeros(self.ruc_every_hours)}, 
            index=self.perturb_timesteps
        )
        for load in self.target_loads:
            cost_allocation[load] = average_shadow_price[load].copy(
                deep=True).rename(columns = {'average_shadow_price' : 'costs'})

            for timestep in cost_allocation[load].index:
                cost_allocation[load].loc[timestep] *= shadow_price_adjustment * \
                    self.act_load_diff.loc[timestep, load]

            allocated_load_costs += cost_allocation[load]

        ## compute allocation for untracked thermal gen
        allocated_thermal_costs = pd.DataFrame(
            {'costs': np.zeros(self.ruc_every_hours)}, 
            index=self.perturb_timesteps
        )
        for gen in self.template_data['ThermalGenerators']:
            load = self.ThermalGeneratorsAtBus[gen]
            cost_allocation[gen] = - average_shadow_price[load].copy(
                deep=True).rename(columns = {'average_shadow_price' : 'costs'})

            for timestep in self.perturb_timesteps:
                cost_allocation[gen].loc[timestep] *= (
                    self.simulations[right]['unpriced_thermal_production'][gen].get(timestep, 0.) - \
                    self.simulations[left]['unpriced_thermal_production'][gen].get(timestep, 0.)
                )

            allocated_thermal_costs += cost_allocation[gen]

        ## compute initial condition costs allocation
        init_costs = []
        for timestep in self.perturb_timesteps:
            costs = 0.
            for gen in self.template_data['ThermalGenerators']:
                ## up
                shadow_price = 0.5 * (self.simulations[left]['shadow_price'][timestep].get(gen + '_Up', 0.) + \
                        self.simulations[right]['shadow_price'][timestep].get(gen + '_Up', 0.))
                costs += shadow_price * (self.simulations[right]['init_thermal_output'][timestep][gen] - \
                            self.simulations[left]['init_thermal_output'][timestep][gen])
                
                ## down
                shadow_price = 0.5 * (self.simulations[left]['shadow_price'][timestep].get(gen + '_Down', 0.) + \
                        self.simulations[right]['shadow_price'][timestep].get(gen + '_Down', 0.))
                costs += shadow_price * (self.simulations[right]['init_thermal_output'][timestep][gen] - \
                            self.simulations[left]['init_thermal_output'][timestep][gen])
                    
            init_costs.append(costs)

        allocation_summary = self.simulations[right]['costs'] - self.simulations[left]['costs']
        print(allocation_summary)
        allocation_summary['total_gen_costs_allocated'] = allocated_gen_costs['costs'].values
        allocation_summary['total_load_costs_allocated'] = allocated_load_costs['costs'].values
        allocation_summary['total_thermal_costs_allocated'] = allocated_thermal_costs['costs'].values
        allocation_summary['initial_costs_allocated'] = init_costs
        allocation_summary['total_costs_allocated'] = allocation_summary['total_gen_costs_allocated'] + \
                allocation_summary['total_load_costs_allocated'] + allocation_summary['initial_costs_allocated'] + \
                allocation_summary['total_thermal_costs_allocated']
        allocation_summary = allocation_summary[['total_cost', 'total_costs_allocated',
                    'total_gen_costs_allocated', 'total_load_costs_allocated', 'total_thermal_costs_allocated', 
                    'initial_costs_allocated']]
        allocation_summary.index.name = 'Time'

        ## compute absolute error threshold if needed
        if self.abs_tol == None:
            df = self.simulations[self.nnods]['costs'] - self.simulations[0]['costs']
            self.abs_tol = self.rel_tol * df['total_cost'].abs().max()

        ## compute allocation error
        err = (allocation_summary['total_cost'] - allocation_summary['total_costs_allocated']).abs().max()
        
        if ( err <= (right - left) / self.nnods * self.abs_tol ) or ( right - left == 1 ):
            self.adaptive_simulations[(left, right)] = {'average_shadow_price' : average_shadow_price,
                                                        'cost_allocation' : cost_allocation,
                                                        'allocation_summary' : allocation_summary}
        else:
            ## recursion
            mid = left + (right - left) // 2
            self.run_adaptive_perturb_analysis(left, mid)
            self.run_adaptive_perturb_analysis(mid, right)


    def compute_allocation(self, adaptive : bool = False):

        if adaptive:
            ## compute cost allocation from adaptive grid

            ## check if adaptive grid covers entire grids
            adaptive_grid = sorted(self.adaptive_simulations.keys(), key = lambda x : x[1])
            if adaptive_grid[0][0] != 0 or adaptive_grid[-1][-1] != self.nnods:
                raise RuntimeError('adaptive grid do not completely cover entire grid')
            for i in range(len(adaptive_grid) - 1):
                if adaptive_grid[i][-1] != adaptive_grid[i + 1][0]:
                    raise RuntimeError('adaptive grid do not completely cover entire grid')

            average_shadow_price = {}
            cost_allocation = {}
            for asset in self.target_assets + self.target_loads:
                average_shadow_price[asset] = reduce(lambda x, y: x.add(y, fill_value=0), 
                    [self.adaptive_simulations[left, right]['average_shadow_price'][asset] * \
                        (right - left) / self.nnods for (left, right) in adaptive_grid])
                cost_allocation[asset] = reduce(lambda x, y: x.add(y, fill_value=0), 
                    [self.adaptive_simulations[nod]['cost_allocation'][asset] for nod in adaptive_grid])

            self.average_shadow_price = average_shadow_price
            self.cost_allocation = cost_allocation
            self.allocation_summary = reduce(lambda x, y: x.add(y, fill_value=0), 
                [self.adaptive_simulations[nod]['allocation_summary'] for nod in adaptive_grid])
            
        else:
            ## compute cost allocation from all grid points

            ## check simulation status at every node
            for nod in range(self.nnods):
                if not self.simulations[nod]['simulated']:
                    raise RuntimeError(
                        'cannot compute cost allocation, run simulation at node {} first'.format(nod))

            average_shadow_price = {}
            cost_allocation = {}

            ## compute average shadow price
            for gen in self.target_assets:
                arr = []
                for timestep in self.perturb_timesteps:
                    p = 0.
                    for nod in range(self.nnods):
                        if nod == 0 or nod == self.nnods:
                            p += 0.5 * self.simulations[nod]['shadow_price'][timestep][gen]
                        else:
                            p += self.simulations[nod]['shadow_price'][timestep][gen]
                    arr.append(p / (self.nnods - 1))
                average_shadow_price[gen] = pd.DataFrame({'average_shadow_price': arr},
                                index = self.effect_timesteps)

            for load in self.target_loads:
                arr = []
                for timestep in self.perturb_timesteps:
                    p = 0.
                    for nod in range(self.nnods):
                        if nod == 0 or nod == self.nnods:
                            p += 0.5 * self.simulations[nod]['shadow_price'][timestep][load]
                        else:
                            p += self.simulations[nod]['shadow_price'][timestep][load]
                    arr.append(p / (self.nnods - 1))
                average_shadow_price[load] = pd.DataFrame({'average_shadow_price': arr},
                                index = self.effect_timesteps)

            ## compute cost allocation
            allocated_gen_costs = pd.DataFrame(
                {'costs': np.zeros(self.ruc_every_hours)}, 
                index=self.perturb_timesteps
            )

            for gen in self.target_assets:
                cost_allocation[gen] = average_shadow_price[gen].copy(
                    deep=True).rename(columns = {'average_shadow_price' : 'costs'})

                for timestep in cost_allocation[gen].index:
                    cost_allocation[gen].loc[timestep] *= self.act_gen_diff.loc[timestep, gen]

                allocated_gen_costs += cost_allocation[gen]

            allocated_load_costs = pd.DataFrame(
                {'costs': np.zeros(self.ruc_every_hours)}, 
                index=self.perturb_timesteps
            )
            for load in self.target_loads:
                cost_allocation[load] = average_shadow_price[load].copy(
                    deep=True).rename(columns = {'average_shadow_price' : 'costs'})

                for timestep in cost_allocation[load].index:
                    cost_allocation[load].loc[timestep] *= self.act_load_diff.loc[timestep, load]

                allocated_load_costs += cost_allocation[load]

            ## compute initial condition costs allocation
            init_costs = []
            for timestep in self.perturb_timesteps:
                costs = 0.
                for nod in range(self.nnods):
                    for gen in self.template_data['ThermalGenerators']:
                        shadow_price = (self.simulations[nod]['shadow_price'][timestep].get(gen + '_Up', 0.) + \
                                self.simulations[nod + 1]['shadow_price'][timestep].get(gen + '_Up', 0.)) / 2.
                        costs += shadow_price * (self.simulations[nod + 1]['init_thermal_output'][timestep][gen] - \
                                    self.simulations[nod]['init_thermal_output'][timestep][gen])
                        
                        shadow_price = (self.simulations[nod]['shadow_price'][timestep].get(gen + '_Down', 0.) + \
                                self.simulations[nod + 1]['shadow_price'][timestep].get(gen + '_Down', 0.)) / 2.
                        costs += shadow_price * (self.simulations[nod + 1]['init_thermal_output'][timestep][gen] - \
                                    self.simulations[nod]['init_thermal_output'][timestep][gen])
                init_costs.append(costs)

            allocation_summary = self.simulations[self.nnods]['costs'] - self.simulations[0]['costs']
            allocation_summary['total_gen_costs_allocated'] = allocated_gen_costs['costs'].values
            allocation_summary['total_load_costs_allocated'] = allocated_load_costs['costs'].values
            allocation_summary['initial_costs_allocated'] = init_costs
            allocation_summary['total_costs_allocated'] = allocation_summary['total_gen_costs_allocated'] + \
                    allocation_summary['total_load_costs_allocated'] + allocation_summary['initial_costs_allocated']
            allocation_summary = allocation_summary[['total_costs', 'total_costs_allocated',
                    'total_gen_costs_allocated', 'total_load_costs_allocated', 'initial_costs_allocated']]

            self.average_shadow_price = average_shadow_price
            self.cost_allocation = cost_allocation
            self.allocation_summary = allocation_summary