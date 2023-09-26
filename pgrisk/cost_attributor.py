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
from copy import deepcopy
from functools import reduce

import gurobipy as gp
from gurobipy import GRB

from .shadow_price import ShadowPriceSimulator


class CostAllocator:
    """
    This abstract class allocates cost of running SCED 
    between the baseline and the target scenarios.
    """

    def __init__(self, 
                template_data: dict,
                start_date: datetime.date, 
                mipgap: float,
                reserve_factor: float, 
                lmp_shortfall_costs: bool,
                init_ruc_file: str | Path | None, 
                verbosity: int,
                target_renewables: list[str], 
                target_loads: list[str],
                baseline_gen_data: pd.DataFrame, 
                baseline_load_data: pd.DataFrame, 
                target_gen_data: pd.DataFrame, 
                target_load_data: pd.DataFrame,
                load_shed_penalty: float,
                reserve_shortfall_penalty: float,
                solver_options: dict, 
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

        self.target_renewables = target_renewables
        self.target_loads = target_loads
        self.baseline_gen_data = baseline_gen_data
        self.baseline_load_data = baseline_load_data
        self.target_gen_data = target_gen_data
        self.target_load_data = target_load_data
        self.load_shed_penalty = load_shed_penalty
        self.reserve_shortfall_penalty = reserve_shortfall_penalty
        self.solver_options = solver_options
        self.scale = scale
        self.tol = tol
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

        ## time step when perturbations affect cost
        self.effect_timesteps = [
            self.start_date.to_pydatetime().replace(tzinfo=None) + timedelta(hours=hour) 
            for hour in range(self.ruc_every_hours)]

        ## get gen and load difference
        self.gen_diff = self.target_gen_data - self.baseline_gen_data
        self.load_diff = self.target_load_data - self.baseline_load_data

        act_gen_diff = self.gen_diff.loc[pd.to_datetime(self.perturb_timesteps, utc=True), 
                                [('actl', asset) for asset in self.target_renewables]].copy(deep=True)
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
        """
        Run simulation at given node and compute shadow price
        """

        simulator = ShadowPriceSimulator(
                load_buses = self.target_loads,
                renewable_gens = self.target_renewables,
                template_data = self.template_data, 
                gen_data = self.baseline_gen_data + self.simulations[nod]['alpha'] * self.gen_diff, 
                load_data = self.baseline_load_data + self.simulations[nod]['alpha'] * self.load_diff,
                out_dir = None, start_date = self.start_date.date(), num_days = 1, 
                solver_options = self.solver_options, run_lmps = False, mipgap = self.mipgap, 
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
                hours_in_objective = 1,
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
        cost_df = simulator.cost_df.copy(deep=True)
        shadow_price = deepcopy(simulator.shadow_price)
        init_thermal_output = deepcopy(simulator.init_thermal_output)
        unpriced_thermal_production = deepcopy(simulator.unpriced_thermal_production)

        if nod == 0:
            self.baseline_report_dfs = report_dfs
        elif nod == self.nnods:
            self.target_report_dfs = report_dfs

        del simulator

        return {'nod' : nod, 'cost' : cost_df, 'shadow_price' : shadow_price, 
            'init_thermal_output' : init_thermal_output,
            'unpriced_thermal_production': unpriced_thermal_production}

    def run_adaptive_perturb_analysis(self, 
            left : int = None, right : int = None) -> None:
        """
        Run integrated gradient on adaptive trapezoidal grid
        """
    
        if left is None:
            left = 0
        if right is None:
            right = self.nnods

        average_shadow_price = {}
        cost_allocation = {}
        
        if not self.simulations[left]['simulated']:
            results = self.simulate(left)
            self.simulations[left]['cost'] = results['cost']
            self.simulations[left]['shadow_price'] = results['shadow_price']
            self.simulations[left]['init_thermal_output'] = results['init_thermal_output']
            self.simulations[left]['simulated'] = True
            self.simulations[left]['unpriced_thermal_production'] = results['unpriced_thermal_production']
            
        if not self.simulations[right]['simulated']:
            results = self.simulate(right)
            self.simulations[right]['cost'] = results['cost']
            self.simulations[right]['shadow_price'] = results['shadow_price']
            self.simulations[right]['init_thermal_output'] = results['init_thermal_output']
            self.simulations[right]['simulated'] = True
            self.simulations[right]['unpriced_thermal_production'] = results['unpriced_thermal_production']
            
        ## compute average shadow price
        for gen in self.target_renewables:
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

        allocated_gen_cost = pd.DataFrame(
            {'cost': np.zeros(self.ruc_every_hours)}, 
            index=self.perturb_timesteps
        )

        for gen in self.target_renewables:
            cost_allocation[gen] = average_shadow_price[gen].copy(
                deep=True).rename(columns = {'average_shadow_price' : 'cost'})

            for timestep in cost_allocation[gen].index:
                cost_allocation[gen].loc[timestep] *= shadow_price_adjustment * \
                    self.act_gen_diff.loc[timestep, gen]

            allocated_gen_cost += cost_allocation[gen]

        ## compute allocation for loads
        allocated_load_cost = pd.DataFrame(
            {'cost': np.zeros(self.ruc_every_hours)}, 
            index=self.perturb_timesteps
        )
        for load in self.target_loads:
            cost_allocation[load] = average_shadow_price[load].copy(
                deep=True).rename(columns = {'average_shadow_price' : 'cost'})

            for timestep in cost_allocation[load].index:
                cost_allocation[load].loc[timestep] *= shadow_price_adjustment * \
                    self.act_load_diff.loc[timestep, load]

            allocated_load_cost += cost_allocation[load]

        ## compute allocation for untracked thermal gen
        allocated_thermal_cost = pd.DataFrame(
            {'cost': np.zeros(self.ruc_every_hours)}, 
            index=self.perturb_timesteps
        )
        for gen in self.template_data['ThermalGenerators']:
            load = self.ThermalGeneratorsAtBus[gen]
            cost_allocation[gen] = - average_shadow_price[load].copy(
                deep=True).rename(columns = {'average_shadow_price' : 'cost'})

            for timestep in self.perturb_timesteps:
                cost_allocation[gen].loc[timestep] *= (
                    self.simulations[right]['unpriced_thermal_production'][gen].get(timestep, 0.) - \
                    self.simulations[left]['unpriced_thermal_production'][gen].get(timestep, 0.)
                )

            allocated_thermal_cost += cost_allocation[gen]

        ## compute initial condition cost allocation
        init_cost = []
        for timestep in self.perturb_timesteps:
            cost = 0.
            for gen in self.template_data['ThermalGenerators']:
                ## up
                shadow_price = 0.5 * (self.simulations[left]['shadow_price'][timestep].get(gen + '_Up', 0.) + \
                        self.simulations[right]['shadow_price'][timestep].get(gen + '_Up', 0.))
                cost += shadow_price * (self.simulations[right]['init_thermal_output'][timestep][gen] - \
                            self.simulations[left]['init_thermal_output'][timestep][gen])
                
                ## down
                shadow_price = 0.5 * (self.simulations[left]['shadow_price'][timestep].get(gen + '_Down', 0.) + \
                        self.simulations[right]['shadow_price'][timestep].get(gen + '_Down', 0.))
                cost += shadow_price * (self.simulations[right]['init_thermal_output'][timestep][gen] - \
                            self.simulations[left]['init_thermal_output'][timestep][gen])
                    
            init_cost.append(cost)

        allocation_summary = self.simulations[right]['cost'] - self.simulations[left]['cost']
        allocation_summary['total_gen_cost_allocated'] = allocated_gen_cost['cost'].values
        allocation_summary['total_load_cost_allocated'] = allocated_load_cost['cost'].values
        allocation_summary['total_thermal_cost_allocated'] = allocated_thermal_cost['cost'].values
        allocation_summary['initial_cost_allocated'] = init_cost
        allocation_summary['total_cost_allocated'] = allocation_summary['total_gen_cost_allocated'] + \
                allocation_summary['total_load_cost_allocated'] + allocation_summary['initial_cost_allocated'] + \
                allocation_summary['total_thermal_cost_allocated']
        allocation_summary = allocation_summary[['total_cost', 'total_cost_allocated',
                    'total_gen_cost_allocated', 'total_load_cost_allocated', 'total_thermal_cost_allocated', 
                    'initial_cost_allocated']]
        allocation_summary.index.name = 'Time'

        ## compute absolute error threshold
        if not hasattr(self, 'abs_tol'):
            df = self.simulations[self.nnods]['cost'] - self.simulations[0]['cost']
            self.abs_tol = self.tol * df['total_cost'].abs().max()

        ## compute allocation error
        err = (allocation_summary['total_cost'] - allocation_summary['total_cost_allocated']).abs().max()
        
        if ( err <= (right - left) / self.nnods * self.abs_tol ) or ( right - left == 1 ):
            self.adaptive_simulations[(left, right)] = {'average_shadow_price' : average_shadow_price,
                                                        'cost_allocation' : cost_allocation,
                                                        'allocation_summary' : allocation_summary}
        else:
            ## recursion
            mid = left + (right - left) // 2
            self.run_adaptive_perturb_analysis(left, mid)
            self.run_adaptive_perturb_analysis(mid, right)

    def compute_allocation(self):
        """
        Compute allocation for individual units
        """
        ## check if grid points covers entire grids
        adaptive_grid = sorted(self.adaptive_simulations.keys(), key = lambda x : x[1])
        if adaptive_grid[0][0] != 0 or adaptive_grid[-1][-1] != self.nnods:
            raise RuntimeError('adaptive grid do not completely cover entire grid')
        for i in range(len(adaptive_grid) - 1):
            if adaptive_grid[i][-1] != adaptive_grid[i + 1][0]:
                raise RuntimeError('adaptive grid do not completely cover entire grid')

        average_shadow_price = {}
        cost_allocation = {}
        for asset in self.target_renewables + self.target_loads:
            average_shadow_price[asset] = reduce(lambda x, y: x.add(y, fill_value=0), 
                [self.adaptive_simulations[left, right]['average_shadow_price'][asset] * \
                    (right - left) / self.nnods for (left, right) in adaptive_grid])
            cost_allocation[asset] = reduce(lambda x, y: x.add(y, fill_value=0), 
                [self.adaptive_simulations[nod]['cost_allocation'][asset] for nod in adaptive_grid])

        self.average_shadow_price = average_shadow_price
        self.cost_allocation = cost_allocation
        self.allocation_summary = reduce(lambda x, y: x.add(y, fill_value=0), 
            [self.adaptive_simulations[nod]['allocation_summary'] for nod in adaptive_grid]).round(2)

