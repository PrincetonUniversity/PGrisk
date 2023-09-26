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
from itertools import product


from gurobipy import GRB, quicksum
from vatic.engines import Simulator

class MarginalSimulatorError(Exception):
    pass

class MarginalSimulator(Simulator):

    def __init__(self, load_buses: list[str] = [], **siml_args) -> None:

        super().__init__(**siml_args)

        self.load_buses = load_buses
        self.thermal_generators = self._data_provider.template['ThermalGenerators']
        self.renewable_generators = self._data_provider.template['NondispatchableGenerators']
        self.marginal_results = dict()

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

        return self._stats_manager.save_output(sim_time)

    def call_oracle(self) -> None:
        """Solve the real-time economic dispatch for the current time step."""

        sced_model = self.solve_sced(hours_in_objective=self.hours_in_objective,
                                     sced_horizon=self.sced_horizon)
        lmps = self.solve_lmp(sced_model) if self.run_lmps else None
        self.marginal_results[self._current_timestep.when] = self.solve_marginal(sced_model)

        self._simulation_state.apply_sced(sced_model)
        self._prior_sced_instance = sced_model

        self._stats_manager.collect_sced_solution(self._current_timestep,
                                                  sced_model, lmps)

    # def solve_marginal(self, sced: ScedModel) -> dict:
    #     """
    #     Gurobi multi-scenarios analysis at each given load bus
    #     with one unit demand increase
    #     """
    #     if sced.model.Status != GRB.OPTIMAL:
    #         raise MarginalSimulatorError(
    #             f"SCED model is not solved to optimality!")

    #     # save the optimal solution
    #     for var in sced.model.getVars():
    #         var._origX = var.X
    #         var.Start = var._origX

    #     # setup multiple-scenario model
    #     sced.model.NumScenarios = len(self.load_buses)

    #     for scenario, bus in enumerate(self.load_buses):
    #         sced.model.Params.ScenarioNumber = scenario

    #         var = sced.model.getVarByName(f"LoadShedding[{bus},1]")
    #         if var:
    #             var.ScenNUB = var.UB + 1.

    #         constr = sced.model.getConstrByName(f"_eq_p_net_withdraw_at_bus[{bus}]_at_period[1]")
    #         if constr:
    #             constr.ScenNRHS = constr.RHS + 1.

    #         var = sced.model.getVarByName(f"ReserveShortfall[1]")
    #         var.ScenNUB = var.UB + self._data_provider._reserve_factor

    #         constr = sced.model.getConstrByName(f"EnforceReserveRequirements[1]")
    #         constr.ScenNRHS = constr.RHS + self._data_provider._reserve_factor

    #         constr = sced.model.getConstrByName(f"eq_p_balance_at_period1")
    #         constr.ScenNRHS = constr.RHS + 1.

    #     sced.model.optimize()

    #     return self.parse_multiple_scenario_model_results(sced)
        

    def solve_marginal(self, sced: ScedModel, mu: float=10) -> dict:
        """
        Gurobi multiple-scenarios analysis at each given load bus
        with one unit demand increase. 
        """
        if sced.model.Status != GRB.OPTIMAL:
            raise MarginalSimulatorError(
                f"SCED model is not solved to optimality!")

        # save the optimal solution
        # for binaries, fix value and relax
        # for var in sced.model.getVars():
        #     var._origX = var.X
        #     if var.VType == GRB.BINARY:
        #         var.lb = var.X
        #         var.ub = var.X
        #         var.setAttr('vtype', 'C')
        #     else:
        #         var.Start = var._origX

        for var in sced.model.getVars():
            var._origX = var.X
            var.Start = var._origX

        # create objective with regularization
        sced.model._PowerGeneratedAboveMinimumDeviationT1 = sced.model.addVars(
            *sced.thermal_periods, lb=0, ub=GRB.INFINITY, 
            vtype=GRB.CONTINUOUS, name='PowerGeneratedAboveMinimumDeviationT1')
        sced.model._PowerGeneratedAboveMinimumDeviationT2 = sced.model.addVars(
            *sced.thermal_periods, lb=0, ub=GRB.INFINITY, 
            vtype=GRB.CONTINUOUS, name='PowerGeneratedAboveMinimumDeviationT2')

        sced.model._RenewablePowerUsedDeviationT1 = sced.model.addVars(
            *sced.renew_periods, lb=0, ub=GRB.INFINITY, 
            vtype=GRB.CONTINUOUS, name='RenewablePowerUsedDeviationT1')
        sced.model._RenewablePowerUsedDeviationT2 = sced.model.addVars(
            *sced.renew_periods, lb=0, ub=GRB.INFINITY, 
            vtype=GRB.CONTINUOUS, name='RenewablePowerUsedDeviationT2')

        power_generated_above_minimum = {
            (g, t): sced.model._PowerGeneratedAboveMinimum[g, t].x
            for g, t in product(*sced.thermal_periods)
        }
        renewable_power_used = {
            (g, t): sced.model._RenewablePowerUsed[g, t].x
            for g, t in product(*sced.renew_periods)
        }

        sced.model._PowerGeneratedAboveMinimumDeviation = sced.model.addConstrs(
            (sced.model._PowerGeneratedAboveMinimumDeviationT1[g, t] - \
                sced.model._PowerGeneratedAboveMinimumDeviationT2[g, t] == \
                sced.model._PowerGeneratedAboveMinimum[g, t] - power_generated_above_minimum[g, t]
             for g, t in product(*sced.thermal_periods)), name='PowerGeneratedAboveMinimumDeviation'
        )
        sced.model._RenewablePowerUsedDeviation = sced.model.addConstrs(
            (sced.model._RenewablePowerUsedDeviationT1[g, t] - \
                sced.model._RenewablePowerUsedDeviationT2[g, t] == \
                sced.model._RenewablePowerUsed[g, t] - renewable_power_used[g, t]
             for g, t in product(*sced.renew_periods)), name='RenewablePowerUsedDeviation'
        )

        sced.model.setObjective(
                sced.model.getObjective() + mu * (
                    quicksum(sced.model._PowerGeneratedAboveMinimumDeviationT1[g, t]
                        for g, t in product(*sced.thermal_periods)) + \
                    quicksum(sced.model._PowerGeneratedAboveMinimumDeviationT2[g, t]
                        for g, t in product(*sced.thermal_periods)) + \
                    quicksum(sced.model._RenewablePowerUsedDeviationT1[g, t]
                        for g, t in product(*sced.renew_periods)) + \
                    quicksum(sced.model._RenewablePowerUsedDeviationT2[g, t]
                        for g, t in product(*sced.renew_periods))
                    ),
                GRB.MINIMIZE
        )

        # setup multiple-scenario model
        sced.model.NumScenarios = len(self.load_buses)

        for scenario, bus in enumerate(self.load_buses):
            sced.model.Params.ScenarioNumber = scenario

            var = sced.model.getVarByName(f"LoadShedding[{bus},1]")
            if var:
                var.ScenNUB = var.UB + 1.

            constr = sced.model.getConstrByName(f"_eq_p_net_withdraw_at_bus[{bus}]_at_period[1]")
            if constr:
                constr.ScenNRHS = constr.RHS + 1.

            var = sced.model.getVarByName(f"ReserveShortfall[1]")
            var.ScenNUB = var.UB + self._data_provider._reserve_factor

            constr = sced.model.getConstrByName(f"EnforceReserveRequirements[1]")
            constr.ScenNRHS = constr.RHS + self._data_provider._reserve_factor

            constr = sced.model.getConstrByName(f"eq_p_balance_at_period1")
            constr.ScenNRHS = constr.RHS + 1.

        # use primal simplex method 0, or dual simplex 1.
        # sced.model.Params.Method = 0
        sced.model.optimize()

        return self.parse_multiple_scenario_model_results(sced)

    def parse_multiple_scenario_model_results(self, sced: ScedModel):

        if sced.model.Status != GRB.OPTIMAL:
            raise MarginalSimulatorError(
                f"Multiple-scenario model is not solved to optimality, "
                f"model status is {sced.model.Status}!")
        if sced.model.NumScenarios != len(self.load_buses):
            raise MarginalSimulatorError(
                f"Multiple-scenario model is not set up properly, "
                f"found only {sced.model.NumScenarios} scenarios!")
            
        results = dict()
        for scenario, bus in enumerate(self.load_buses):
            sced.model.Params.ScenarioNumber = scenario 
            if sced.model.ScenNObjVal < GRB.INFINITY:
                # verify regularization is valid
                for g, t in product(*sced.thermal_periods):
                    t1 = sced.model._PowerGeneratedAboveMinimumDeviationT1[g, t].x
                    t2 = sced.model._PowerGeneratedAboveMinimumDeviationT2[g, t].x
                    if t1 != 0 and t2 != 0:
                        raise MarginalSimulatorError(f"invalid regularization parameters "
                                                     f"{t1} and {t2} for {g} at time {t}!")
                
                for g, t in product(*sced.renew_periods):
                    t1 = sced.model._RenewablePowerUsedDeviationT1[g, t].x
                    t2 = sced.model._RenewablePowerUsedDeviationT2[g, t].x
                    if t1 != 0 and t2 != 0:
                        raise MarginalSimulatorError(f"invalid regularization parameters "
                                                     f"{t1} and {t2} for {g} at time {t}!")

                at_bus = dict()
                for gen in sced.ThermalGenerators:
                    orig_output = sced.model.getVarByName(f"PowerGeneratedAboveMinimum[{gen},1]")._origX + \
                            sced.model.getVarByName(f"UnitOn[{gen},1]")._origX * sced.MinPowerOutput[gen, 1]
                    scen_output = sced.model.getVarByName(f"PowerGeneratedAboveMinimum[{gen},1]").ScenNX + \
                            sced.model.getVarByName(f"UnitOn[{gen},1]").ScenNX * sced.MinPowerOutput[gen, 1]
                    diff_output = scen_output - orig_output

                    if abs(diff_output) > 1 / 10 ** self._stats_manager.max_decimals:
                        at_bus[gen] = {
                            'Original': orig_output, 
                            'Marginal': scen_output,
                            'Deviation': diff_output,
                        }


                for gen in sced.RenewableGenerators:
                    orig_output = sced.model.getVarByName(f"NondispatchablePowerUsed[{gen},1]")._origX
                    scen_output = sced.model.getVarByName(f"NondispatchablePowerUsed[{gen},1]").ScenNX
                    diff_output = scen_output - orig_output

                    if abs(diff_output) > 1 / 10 ** self._stats_manager.max_decimals:
                        at_bus[gen] = {
                            'Original': orig_output, 
                            'Marginal': scen_output,
                            'Deviation': diff_output,
                        }

                df = pd.DataFrame.from_dict(at_bus, orient='index').round(self._stats_manager.max_decimals)
                if at_bus:
                    df.loc["Total"] = df.sum()
                results[bus] = {'objective': sced.model.ScenNObjVal, 'df': df}
            else:
                raise MarginalSimulatorError(f"Marginal scenario for bus {bus} is not solved to optimality!")            

        return results

