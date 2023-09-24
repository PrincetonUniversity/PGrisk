from __future__ import annotations

from typing import Union
from pathlib import Path
import pandas as pd
import os
import time
from datetime import datetime, timedelta
import shutil
import gurobipy as gp
from gurobipy import GRB
from vatic.engines import Simulator

class ShadowPriceError(Exception):
    pass

class ShadowPriceSimulator(Simulator):

    def __init__(self, 
            load_buses: list[str], renewable_gens: list[str], 
            workdir: Union[Path, str], **siml_args) -> None:

        super().__init__(**siml_args)

        if self.sced_horizon != 1:
            raise ShadowPriceError("SCED horizon must be set to 1 for shadow price simulations!")

        self.load_buses = load_buses
        self.renewable_gens = renewable_gens
        self.mipgap = siml_args['mipgap']
        self.thermal_generators = self._data_provider.template['ThermalGenerators']
        self.thermal_pmin = self._data_provider.template['MinimumPowerOutput']
        self.thermal_pmax = self._data_provider.template['MaximumPowerOutput']

        self.NondispatchableGeneratorsAtBus = dict()
        for bus in self._data_provider.template['NondispatchableGeneratorsAtBus']:
            if self._data_provider.template['NondispatchableGeneratorsAtBus'][bus]:
                for gen in self._data_provider.template['NondispatchableGeneratorsAtBus'][bus]:
                    self.NondispatchableGeneratorsAtBus[gen] = bus

        ## working directory
        self.workdir = Path(workdir)
        os.makedirs(self.workdir, exist_ok=True)

        self.report_dfs = None
        self.init_thermal_production = dict()
        self.shadow_price = dict()
        self.unpriced_thermal_production = dict()

    def clean_wkdir(self):
        shutil.rmtree(self.workdir)

    def solve_save_sced(self,
                   hours_in_objective: int,
                   sced_horizon: int) -> VaticModelData:

        """This method is adapted from engine.solve_sced. 
        SCED model is saved to MPS file.
        """

        sced_model_data = self._data_provider.create_sced_instance(
            self._simulation_state, sced_horizon=sced_horizon)

        ## get thermal generators initial output level
        gen_data = sced_model_data._data['elements']['generator']
        self.init_thermal_production[self._current_timestep.when] = {
            gen: gen_data[gen]['initial_p_output']
            for gen in gen_data if gen_data[gen]['generator_type'] != 'renewable'
        }

        self._ptdf_manager.mark_active(sced_model_data)

        self._hours_in_objective = hours_in_objective
        if self._hours_in_objective > 10:
            ptdf_options = self._ptdf_manager.look_ahead_sced_ptdf_options
        else:
            ptdf_options = self._ptdf_manager.sced_ptdf_options

        self.sced_model.generate_model(
            sced_model_data, relax_binaries=False, ptdf_options=ptdf_options,
            ptdf_matrix_dict=self._ptdf_manager.PTDF_matrix_dict,
            objective_hours=hours_in_objective
            )
        
        # update in case lines were taken out
        self._ptdf_manager.PTDF_matrix_dict = self.sced_model.pyo_instance._PTDFs

        sced_results = self.sced_model.solve_model(self._sced_solver,
                                                   self.solver_options)
        
        ## save sced model instance to files
        model = self.sced_model.pyo_instance
        filename = 'sced_' + self._current_timestep.when.strftime('%H%M') + '.mps'
        model.write(str(self.workdir / filename), io_options={'symbolic_solver_labels': True})

        self._ptdf_manager.update_active(sced_results)

        return sced_results

    def call_save_oracle(self) -> None:
        """Solves the real-time economic dispatch and save sced optimization to file.
        This method is adapted from engine.call_oracle.
        """

        if self.verbosity > 0:
            print("\nSolving SCED instance")

        current_sced_instance = self.solve_save_sced(hours_in_objective=1,
                                                sced_horizon=self.sced_horizon)

        if self.verbosity > 0:
            print("Solving for LMPs")

        if self.run_lmps:
            lmp_sced = self.solve_lmp(current_sced_instance)
        else:
            lmp_sced = None

        self._simulation_state.apply_sced(current_sced_instance)
        self._prior_sced_instance = current_sced_instance

        self._stats_manager.collect_sced_solution(
            self._current_timestep, current_sced_instance, lmp_sced,
            pre_quickstart_cache=None
            )

    def simulate(self) -> dict[str, pd.DataFrame]:
        """Top-level runner of a simulation.
        """
        simulation_start_time = time.time()

        # create commitments for the first day using an RUC
        self.initialize_oracle()
        self.simulation_times['Init'] += time.time() - simulation_start_time

        for time_step in self._time_manager.time_steps():
            self._current_timestep = time_step

            # run the day-ahead RUC at some point in the day before
            if time_step.is_planning_time:
                plan_start_time = time.time()

                self.call_planning_oracle()
                self.simulation_times['Plan'] += time.time() - plan_start_time

            # run the SCED to simulate this time step
            oracle_start_time = time.time()
            self.call_save_oracle()
            self.simulation_times['Sim'] += time.time() - oracle_start_time

        sim_time = time.time() - simulation_start_time

        if self.verbosity > 0:
            print("Simulation Complete")
            print("Total simulation time: {:.1f} seconds".format(sim_time))

            if self.verbosity > 1:
                print("Initialization time: {:.2f} seconds".format(
                    self.simulation_times['Init']))
                print("Planning time: {:.2f} seconds".format(
                    self.simulation_times['Plan']))
                print("Real-time sim time: {:.2f} seconds".format(
                    self.simulation_times['Sim']))

        self.report_dfs = self._stats_manager.save_output(sim_time)

        ## get thermal outputs at each timestep
        df = self.report_dfs['thermal_detail'].copy(deep=True)
        df['Datetime'] = (pd.to_datetime(df.index.get_level_values(0)) + \
                        pd.to_timedelta(df.index.get_level_values(1), unit='H')).to_pydatetime()
        gb = df.reset_index().groupby('Datetime')
        thermal_outputs = {datetime.to_pydatetime() : dict(zip(df.Generator, df.Dispatch)) for datetime, df in gb}
        self.thermal_outputs = thermal_outputs

        cost_df = pd.DataFrame.from_dict(self._stats_manager._sced_stats, orient='index',
                columns=['fixed_costs', 'variable_costs', 'reserve_shortfall', 'load_shedding', 'over_generation'])

        cost_df['total_costs'] = cost_df['fixed_costs'] + cost_df['variable_costs'] + \
            self._data_provider._load_mismatch_cost * (cost_df['load_shedding'] + cost_df['over_generation']) + \
            self._data_provider._reserve_mismatch_cost * cost_df['reserve_shortfall']
        cost_df.index = cost_df.index.map(lambda x: x.when)

        unpriced_thermal_production = {gen : {} for gen in self.thermal_generators}
        fixed_cost_adjustments, variable_cost_adjustments = [], []
        for timestep in self._stats_manager._sced_stats:
            sced_stats = self._stats_manager._sced_stats[timestep]

            for th_gen in sced_stats['observed_thermal_states']:
                if (not sced_stats['observed_thermal_states'][th_gen]) and \
                    (sced_stats['observed_thermal_dispatch_levels'][th_gen] > 0.):
                    unpriced_thermal_production[th_gen][timestep.when] = sced_stats['observed_thermal_dispatch_levels'][th_gen]

        self.unpriced_thermal_production = unpriced_thermal_production
        self.cost_df = cost_df


    def simulate_shadow_price(self) -> None:
        """simulate and get shadow price by computing the constraint dual value
        """

        env = gp.Env(params = {"LogToConsole":  0})
        shadow_price = {}

        for current_timestep in self._time_manager.time_steps():

            self._current_timestep = current_timestep
            
            ## solve model and store binary optimal values
            filename = 'sced_' + self._current_timestep.when.strftime('%H%M') + '.mps'
            gurobi_md = gp.read(str(self.workdir / filename), env)
            gurobi_md.Params.LogToConsole = 0
            gurobi_md.setParam('MIPGap', self.mipgap)
            gurobi_md.optimize()

            binary_optimal_vals = {
            var.VarName: var.X for var in gurobi_md.getVars() \
                if (var.LB == 0. and var.UB == 1.) and var.VType in (GRB.BINARY, GRB.INTEGER)
            }

            ## reload the model
            filename = 'sced_' + self._current_timestep.when.strftime('%H%M') + '.mps'
            gurobi_md = gp.read(str(self.workdir / filename), env)
            gurobi_md.Params.LogToConsole = 0

            for var in gurobi_md.getVars():
                ##  fix binaries and reduce MIP to LP
                if (var.LB == 0. and var.UB == 1.) and var.VType in (GRB.BINARY, GRB.INTEGER):
                    var.setAttr('vtype', 'C')
                    var.LB = var.UB = binary_optimal_vals[var.VarName]

            ## add renewable production UB as constraints to compute dual value
            for gen in self.renewable_gens:
                varname = 'NondispatchablePowerUsed(' + \
                    gen.replace('-', '_').replace('/', '') + '_1)'
                var = gurobi_md.getVarByName(varname)

                if var.LB > 0:
                    if var.LB == var.UB:
                        ## nondispatchable renewable pmin = pmax
                        gurobi_md.addLConstr(var == var.UB, var.VarName + '_UB')
                        var.LB = -GRB.INFINITY
                        var.UB = GRB.INFINITY
                    else:
                        raise RuntimeError('nondispatchable renewable shoud have have pmin equals to pmax!')
                else:
                    ## dispatchable renewable pmin = 0.
                    gurobi_md.addLConstr(var <= var.UB, var.VarName + '_UB')
                    var.UB = GRB.INFINITY

            ## add loadshedding and overgeneration UBs as constraints to compute dual value
            for bus in self.load_buses:
                varname = 'LoadShedding(' + \
                    bus.replace('-', '_').replace('/', '') + '_1)'
                var = gurobi_md.getVarByName(varname)

                if var:
                    gurobi_md.addLConstr(var <= var.UB, var.VarName + '_UB')
                    var.UB = GRB.INFINITY

                varname = 'OverGeneration(' + \
                    bus.replace('-', '_').replace('/', '') + '_1)'
                var = gurobi_md.getVarByName(varname)

                if var:
                    gurobi_md.addLConstr(var <= var.UB, var.VarName + '_UB')
                    var.UB = GRB.INFINITY

            ## add reserve shortfall UB as constraint to compute dual value
            var = gurobi_md.getVarByName('ReserveShortfall(1)')
            gurobi_md.addLConstr(var <= var.UB, var.VarName + '_UB')
            var.UB = GRB.INFINITY
                
            gurobi_md.setParam('MIPGap', self.mipgap)
            gurobi_md.optimize()   

            ## get shadow price for renewables
            shadow_price[self._current_timestep.when] = {}

            for gen in self.renewable_gens:
                constrname = 'NondispatchablePowerUsed(' + \
                    gen.replace('-', '_').replace('/', '') + '_1)_UB'
                constr = gurobi_md.getConstrByName(constrname)
                shadow_price[self._current_timestep.when][gen] = constr.Pi

                bus = self.NondispatchableGeneratorsAtBus[gen]

                constr = gurobi_md.getConstrByName('OverGeneration(' + \
                    bus.replace('-', '_').replace('/', '') + '_1)_UB')
                if constr:
                    shadow_price[self._current_timestep.when][gen] += constr.Pi

            for gen in self.thermal_generators:
                constrname = 'c_u_EnforceMaxAvailableRampUpRates(' + \
                    gen.replace('-', '_').replace('/', '') + '_1)_'
                constr = gurobi_md.getConstrByName(constrname)
                if constr:
                    shadow_price[self._current_timestep.when][gen + '_Up'] = constr.Pi

                constrname = 'c_l_EnforceScaledNominalRampDownLimits(' + \
                    gen.replace('-', '_').replace('/', '') + '_1)_'
                constr = gurobi_md.getConstrByName(constrname)
                if constr:
                    shadow_price[self._current_timestep.when][gen + '_Down'] = constr.Pi

            ## get shadow price for loads
            constr = gurobi_md.getConstrByName('c_e_TransmissionBlock(1)_eq_p_balance_')
            eq_p_balance = constr.Pi

            constr = gurobi_md.getConstrByName('c_l_EnforceReserveRequirements(1)_')
            enforece_reserve = constr.Pi

            constr = gurobi_md.getConstrByName('ReserveShortfall(1)_UB')
            reserve_ub = constr.Pi

            load_reference_shadow_price = eq_p_balance + \
                            self._data_provider._reserve_factor * (enforece_reserve + reserve_ub)

            for bus in self.load_buses:
                shadow_price[self._current_timestep.when][bus] = load_reference_shadow_price

                constr = gurobi_md.getConstrByName('c_e_TransmissionBlock(1)_eq_p_net_withdraw_at_bus(' + \
                     bus.replace('-', '_').replace('/', '') + ')_')
                if constr:
                    shadow_price[self._current_timestep.when][bus] += constr.Pi
                
                constr = gurobi_md.getConstrByName('LoadShedding(' + \
                    bus.replace('-', '_').replace('/', '') + '_1)_UB')
                if constr:
                    shadow_price[self._current_timestep.when][bus] += constr.Pi

                constr = gurobi_md.getConstrByName('OverGeneration(' + \
                    bus.replace('-', '_').replace('/', '') + '_1)_UB')
                if constr:
                    shadow_price[self._current_timestep.when][bus] += constr.Pi

        self.shadow_price = shadow_price

