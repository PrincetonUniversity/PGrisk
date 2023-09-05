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

from vatic.engines import Simulator

class ShadowPriceError(Exception):
    pass

class ShadowPriceSimulator(Simulator):

    def __init__(self, load_buses: list[str] = [], 
            renewable_gens: list[str] = [], **siml_args) -> None:

        super().__init__(**siml_args)

        if self.sced_horizon != 1:
            raise ShadowPriceError("SCED horizon must be set to 1 for shadow price simulations!")

        self.load_buses = load_buses
        self.renewable_gens = renewable_gens
        self.thermal_generators = self._data_provider.template['ThermalGenerators']
        self.thermal_pmin = self._data_provider.template['MinimumPowerOutput']
        self.thermal_pmax = self._data_provider.template['MaximumPowerOutput']

        self.ScaledNominalRampUpLimit = {
            gen : min(self._data_provider.template['NominalRampUpLimit'][gen], 
                self._data_provider.template['MaximumPowerOutput'][gen]) for gen 
                in self._data_provider.template['ThermalGenerators']
        }

        self.ScaledNominalRampDownLimit = self._data_provider.template['NominalRampDownLimit']
        self.ShutdownRampLimit = self._data_provider.template['ShutdownRampLimit']

        self.NondispatchableGeneratorsAtBus = dict()
        for bus in self._data_provider.template['NondispatchableGeneratorsAtBus']:
            if self._data_provider.template['NondispatchableGeneratorsAtBus'][bus]:
                for gen in self._data_provider.template['NondispatchableGeneratorsAtBus'][bus]:
                    self.NondispatchableGeneratorsAtBus[gen] = bus

        self.report_dfs = None
        self.init_thermal_output = dict()
        self.commits = dict()
        self.shadow_price = dict()
        self.perturb_costs = dict()
        self.EnforceMaxAvailableRampUpRates = dict()
        self.EnforceScaledNominalRampDownLimits = dict()
        self.unpriced_thermal_production = dict()

    def simulate(self) -> dict[str, pd.DataFrame]:
        """Top-level runner of a simulation's alternating RUCs and SCEDs.

        See prescient.simulator.Simulator.simulate() for the original
        implementation of this logic.

        """
        simulation_start_time = time.time()

        # create commitments for the first day using an initial unit commitment
        self.initialize_oracle()
        self.simulation_times['Init'] += time.time() - simulation_start_time

        # simulate each time period
        for time_step in self._time_manager.time_steps():
            self._current_timestep = time_step

            # run the day-ahead RUC at some point in the day before
            if time_step.is_planning_time:
                plan_start_time = time.time()

                self.call_planning_oracle()
                self.simulation_times['Plan'] += time.time() - plan_start_time

            # run the SCED to simulate this time step
            oracle_start_time = time.time()
            self.call_oracle()
            self.simulation_times['Sim'] += time.time() - oracle_start_time

        sim_time = time.time() - simulation_start_time

        print("Simulation Complete")
        print("Total simulation time: {:.1f} seconds".format(sim_time))

        print("Initialization time: {:.2f} seconds".format(
            self.simulation_times['Init']))
        print("Planning time: {:.2f} seconds".format(
            self.simulation_times['Plan']))
        print("Real-time sim time: {:.2f} seconds".format(
            self.simulation_times['Sim']))

        report_dfs = self._stats_manager.save_output(sim_time)

        ## get thermal outputs at each timestep
        df = report_dfs['thermal_detail'].copy(deep=True)
        df['Datetime'] = (pd.to_datetime(df.index.get_level_values(0)) + \
                        pd.to_timedelta(df.index.get_level_values(1), unit='H')).to_pydatetime()
        gb = df.reset_index().groupby('Datetime')
        thermal_outputs = {datetime.to_pydatetime() : dict(zip(df.Generator, df.Dispatch)) for datetime, df in gb}
        self.thermal_outputs = thermal_outputs

        cost_df = pd.DataFrame.from_dict(self._stats_manager._sced_stats, orient='index',
                columns=['fixed_cost', 'variable_cost', 'reserve_shortfall', 'load_shedding', 'over_generation'])

        cost_df['total_cost'] = cost_df['fixed_cost'] + cost_df['variable_cost'] + \
            self._data_provider._load_mismatch_cost * (cost_df['load_shedding'] + cost_df['over_generation']) + \
            self._data_provider._reserve_mismatch_cost * cost_df['reserve_shortfall']
        cost_df.index = cost_df.index.map(lambda x: x.when)

        unpriced_thermal_production = {gen : {} for gen in self.thermal_generators}
        fixed_cost_adjustments, variable_cost_adjustments = [], []
        for timestep in self._stats_manager._sced_stats:
            sced_stats = self._stats_manager._sced_stats[timestep]

            for th_gen, unit_on in sced_stats['observed_thermal_states'].items():
                if not unit_on and sced_stats['observed_thermal_dispatch_levels'][th_gen] > 0.:
                    unpriced_thermal_production[th_gen][timestep.when] = sced_stats['observed_thermal_dispatch_levels'][th_gen]

        self.unpriced_thermal_production = unpriced_thermal_production
        self.cost_df = cost_df

        return report_dfs

    def call_oracle(self) -> None:
        """Solve the real-time economic dispatch for the current time step."""

        sced_model = self.solve_sced(hours_in_objective=self.sced_horizon,
                                     sced_horizon=self.sced_horizon)

        # save thermal output
        self.init_thermal_output[self._current_timestep.when] = deepcopy(sced_model.PowerGeneratedT0)
        
        shadow_price = self.solve_shadow_price(sced_model)
        lmps = self.solve_lmp(sced_model) if self.run_lmps else None

        self._simulation_state.apply_sced(sced_model)
        self._prior_sced_instance = sced_model

        self._stats_manager.collect_sced_solution(self._current_timestep,
                                                  sced_model, lmps)

    def solve_shadow_price(self, sced: ScedModel) -> dict:

        # relax binaries and fix their values
        if sced.model.Status != 2:
            raise VaticModelError(f"model must be solved to optimality "
                                   "before fixing and relaxing binaries!")
        for varray in sced.binary_variables:
            for var in varray.values():
                var.setAttr('vtype', 'C')
                var.LB = var.X
                var.UB = var.X
        sced.model.update()

        sced.add_objective()

        ## add renewable production UB as constraints
        for gen in self.renewable_gens:
            var_name = f"NondispatchablePowerUsed[{gen},1]"
            constr_name = f"NondispatchablePowerUsedUpperBound[{gen},1]"
            var = sced.model.getVarByName(var_name)
            if var.LB > 0:
                if var.LB == var.UB:
                    ## nondispatchable renewable pmin = pmax
                    sced.model.addLConstr(var == var.UB, constr_name)
                    var.LB = -GRB.INFINITY
                    var.UB = GRB.INFINITY
                else:
                    raise ShadowPriceError(f"nondispatchable renewable must have pmin equal to pmax!")
            else:
                ## dispatchable renewable pmin = 0.
                sced.model.addLConstr(var <= var.UB, constr_name)
                var.UB = GRB.INFINITY 

        ## add loadshedding and overgeneration UBs as constraints to compute dual value
        for bus in self.load_buses:
            var_name = f"LoadShedding[{bus},1]"
            constr_name = f"LoadSheddingUpperBound[{bus},1]"
            var = sced.model.getVarByName(var_name)
            if var:
                sced.model.addLConstr(var <= var.UB, constr_name)
                var.UB = GRB.INFINITY

            var_name = f"OverGeneration[{bus},1]"
            constr_name = f"OverGenerationUpperBound[{bus},1]"
            var = sced.model.getVarByName(var_name)
            if var:
                sced.model.addLConstr(var <= var.UB, constr_name)
                var.UB = GRB.INFINITY

        ## add reserve shortfall UB as constraint to compute dual value
        var = sced.model.getVarByName(f"ReserveShortfall[1]")
        sced.model.addLConstr(var <= var.UB, f"ReserveShortfallUpperBound[1]")
        var.UB = GRB.INFINITY

        # solve LP
        sced.model.Params.OutputFlag = 0
        sced.model.Params.MIPGap = self.mipgap
        sced.model.Params.Threads = self.solver_options['Threads']
        sced.model.optimize()

        # compute shadow price for renewable gens
        self.shadow_price[self._current_timestep.when] = {}

        for gen in self.renewable_gens:
            constr_name = f"NondispatchablePowerUsedUpperBound[{gen},1]"
            constr = sced.model.getConstrByName(constr_name)
            self.shadow_price[self._current_timestep.when][gen] = constr.Pi

            bus = self.NondispatchableGeneratorsAtBus[gen]

            cosntr_name = f"OverGenerationUpperBound[{gen},1]"
            constr = sced.model.getConstrByName(cosntr_name)
            if constr:
                self.shadow_price[self._current_timestep.when][gen] += constr.Pi

        # compute shadow price for initial state of thermal gens
        for gen in self.thermal_generators:
            constr_name = f"enforce_max_available_tk_ramp_up_rates[{gen},1]"
            constr = sced.model.getConstrByName(constr_name)
            if constr:
                self.shadow_price[self._current_timestep.when][f"{gen}_Up"] = constr.Pi

            constr_name = f"enforce_max_available_tk_ramp_down_rates[{gen},1]"
            constr = sced.model.getConstrByName(constr_name)
            if constr:
                self.shadow_price[self._current_timestep.when][f"{gen}_Down"] = constr.Pi

        ## get shadow price for loads
        constr = sced.model.getConstrByName(f"eq_p_balance_at_period1")
        eq_p_balance = constr.Pi

        constr = sced.model.getConstrByName(f"EnforceReserveRequirements[1]")
        enforece_reserve = constr.Pi

        constr = sced.model.getConstrByName(f"ReserveShortfallUpperBound[1]")
        reserve_ub = constr.Pi

        load_reference_shadow_price = eq_p_balance + \
                        self._data_provider._reserve_factor * (enforece_reserve + reserve_ub)

        for bus in self.load_buses:
            self.shadow_price[self._current_timestep.when][bus] = load_reference_shadow_price

            constr = sced.model.getConstrByName(f"_eq_p_net_withdraw_at_bus[{bus}]_at_period[1]")
            if constr:
                self.shadow_price[self._current_timestep.when][bus] += constr.Pi
            
            constr = sced.model.getConstrByName(f"LoadSheddingUpperBound[{bus},1]")
            if constr:
                self.shadow_price[self._current_timestep.when][bus] += constr.Pi

            constr = sced.model.getConstrByName(f"OverGenerationUpBound[{bus},1]")
            if constr:
                self.shadow_price[self._current_timestep.when][bus] += constr.Pi

        # return the SCED to its original state
        sced.enforce_binaries()
        sced.add_objective()

