import pandas as pd
import numpy as np
from functools import reduce

from pathlib import Path
import dill as pickle
import os
from datetime import datetime
import sys
import argparse
from vatic.data.loaders import load_input, RtsLoader, T7kLoader


def compute_cost(p, break_points, prices):

    if len(break_points) != len(prices) + 1:
        raise RuntimeError('break points and price do not match')
        
    if p < break_points[0] or p > break_points[-1]:
        raise(RuntimeError('Power production outside of break points'))
        
    cost = 0.
    for i in range(1, len(break_points)):
        if break_points[i - 1] <= p <= break_points[i]:
            return cost + prices[i - 1] * (p - break_points[i - 1])
        else:
            cost += prices[i - 1] * (break_points[i] - break_points[i - 1])
            
# def get_renewable_cost_curve(fcst: float, total_risk: float, scens: np.array, 
#                              risky_scens: np.array,
#                              pmin: float, pmax: float, 
#                              num_of_break_points: int=5):
#     """
#     Compute a cost curve of renewable generator using ECDF
#     """
#     if num_of_break_points <= 0:
#         raise(RuntimeError('num_of_break_points must be positive'))
        
#     if total_risk < 0.:
#         raise(RuntimeError('total_risk must be positive'))
        
#     if fcst < pmin or fcst > pmax:
#         print('fcst must be between pmin and pmax')
#         print(pmin, fcst, pmax)
#         fcst = np.clip(fcst, pmin, pmax)
        
    
#     scens = np.clip(scens, pmin, pmax)
#     scens = sorted(scens)
    
#     # Compute break points
#     scen_min, scen_max = np.min(scens), np.max(scens)
#     if scen_min > pmin:
#         num_of_break_points -= 1

#     if num_of_break_points < 2:
#         raise(RuntimeError('try to increase num_of_break_points'))
        
#     if fcst > scen_min:
#         stepsize = (fcst - scen_min) / (num_of_break_points - 1)
#     else:
#         stepsize = (scen_max - scen_min) / (num_of_break_points - 1)
        
#     # if stepsize == 0 then all scenarios == 0, return zero cost
#     if np.isclose(stepsize, 0., atol=1e-4):
#         return [0.], [pmin, pmax], [0., 0.]
        
#     break_points = list(np.arange(scen_min - 1e-3, scen_max, step=stepsize))  
#     break_points.append(scen_max)
    
#     if pmin < break_points[0]:
#         break_points = [pmin] + break_points
#     if pmax > break_points[-1]:
#         break_points = break_points + [pmax]
                                      
#     df = pd.DataFrame({'Scen': scens, 'Bin': pd.cut(scens, break_points, include_lowest=True)})
    
#     counts = df.groupby('Bin').count()
#     prices = counts.cumsum()['Scen'].tolist()
    
#     fcst_cost = compute_cost(fcst, break_points, prices)
#     risky_costs = [fcst_cost - compute_cost(p, break_points, prices) for p in risky_scens]
    
#     if np.isclose(np.mean(risky_costs), 0.) or np.mean(risky_costs) < 0.:
#         return [0.], [pmin, pmax], [0., 0.]
        
#     # Rescale prices to match total risk
#     prices = list(np.array(prices) * total_risk / np.mean(risky_costs))
    
#     # Adjust price for the last bin
#     if scen_max < pmax:
#         prices[-1] = 1 / np.finfo(np.float64).resolution
        
#     # Adjust boundaries of break points
#     if break_points[1] == 0.:
#         break_points = break_points[1:]
#         prices = prices[1:]
#     else:
#         break_points[0] = max(0., break_points[0])
    
#     if break_points[-2] == pmax:
#         break_points = break_points[0:-1]
#         prices = prices[0:-1]
#     else:
#         break_points[-1] = min(pmax, break_points[-1])
    
#     ## get marginal costs
#     assert len(break_points) == len(prices) + 1, 'size of break_points and prices do not match'
    
#     marginal_prices = [0.]
#     for i in range(len(prices)):
#         marginal_prices.append(marginal_prices[-1] + prices[i] * (break_points[i + 1] - break_points[i]))
    
#     return prices, break_points, marginal_prices

## New version. Updated on 08/29/2023
def get_renewable_cost_curve(fcst: float, total_risk: float, scens: np.array, 
                             risky_scens: np.array,
                             pmin: float, pmax: float, 
                             num_of_break_points: int=5,
                             max_cost: int=1e6):
    """
    Compute a cost curve of renewable generator using ECDF
    """
    if num_of_break_points <= 0:
        raise RuntimeError('num_of_break_points must be positive')
        
    if total_risk < 0.:
        raise RuntimeError('total_risk must be positive')
        
    if fcst < pmin or fcst > pmax:
        print('fcst must be between pmin and pmax')
        print(pmin, fcst, pmax)
        fcst = np.clip(fcst, pmin, pmax)
        
    scens = np.clip(scens, pmin, pmax)
    scens = sorted(scens)
    
    # Compute break points
    scen_min, scen_max = scens[0], scens[-1]
    if scen_min > pmin:
        num_of_break_points -= 1

    if num_of_break_points < 2:
        raise(RuntimeError('try to increase num_of_break_points'))
        
    stepsize = (fcst - scen_min) / (num_of_break_points - 1)
        
    # if stepsize == 0 then all scenarios == 0, return zero cost
    if np.isclose(stepsize, 0., atol=1e-2):
        return [0.], [pmin, pmax], [0., 0.]
        
    break_points = list(np.arange(scen_min, scen_max, step=stepsize))
    if break_points[-1] != scen_max:
        break_points.append(scen_max)
    bins = [p for p in break_points]
        
    if pmin < break_points[0]:
        break_points = [pmin] + break_points
        bins[0] -= (bins[0] - pmin) / 2
        bins = [pmin] + bins
    if pmax > break_points[-1]:
        break_points = break_points + [pmax]
        bins = bins + [pmax]

    df = pd.DataFrame({'Scen': scens, 'Bin': pd.cut(scens, bins, include_lowest=True)})
    
    counts = df.groupby('Bin').count()
    prices = counts.cumsum()['Scen'].tolist()
    
    fcst_cost = compute_cost(fcst, break_points, prices)
    risky_costs = [fcst_cost - compute_cost(p, break_points, prices) for p in risky_scens]
    
    if np.isclose(np.mean(risky_costs), 0.) or np.mean(risky_costs) < 0.:
        return [0.], [pmin, pmax], [0., 0.]
        
    # Rescale prices to match total risk
    prices = list(np.array(prices) * total_risk / np.mean(risky_costs))
    
    # Adjust price for the last bin
    if scen_max < pmax:
        prices[-1] = max_cost
        
    # Adjust boundaries of break points
    if break_points[1] == 0.:
        break_points = break_points[1:]
        prices = prices[1:]
    else:
        break_points[0] = max(0., break_points[0])
    
    if break_points[-2] == pmax:
        break_points = break_points[0:-1]
        prices = prices[0:-1]
    else:
        break_points[-1] = min(pmax, break_points[-1])
    
    ## get marginal costs
    assert len(break_points) == len(prices) + 1, 'size of break_points and prices do not match'
    
    marginal_prices = [0.]
    for i in range(len(prices)):
        marginal_prices.append(marginal_prices[-1] + prices[i] * (break_points[i + 1] - break_points[i]))
    
    return prices, break_points, marginal_prices


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("date", type=str,
                        help="which days to simulate")

    parser.add_argument("grid", type=str, choices={'RTS-GMLC', 'Texas-7k'},
                        help="which power grid to simulate over")

    parser.add_argument("risk_type", type=str, choices={'daily', 'hourly'},
                        help="type of risk, daily or hourly")  

    parser.add_argument("nscens",
                        type=int, default=1000,
                        help="number of scenarios to be simulated")

    parser.add_argument("alpha",
                        type=float, default=0.05,
                        help="percentage of scenarios considered to be risky")

    parser.add_argument("scen_dir", type=str, help="scenario directory")
    parser.add_argument("cost_alloc_dir", type=str, help="cost allocation directory")
    parser.add_argument("reserve_sim_dir", type=str, help="reserve simulation directory")
    parser.add_argument("cost_curve_dir", type=str, help="cost curve directory")
    
    args = parser.parse_args()
    
    start_date = pd.to_datetime(args.date, utc=True)

    template, gen_data, load_data = load_input(args.grid, 
        args.date, num_days=1, init_state_file=None)

    ## get scenario data
    if args.grid == 'RTS-GMLC':
        loader = RtsLoader()
    elif args.grid == 'Texas-7k':
        loader = T7kLoader()
    else:
        raise ValueError("unrecognized grid name")

    scen_dfs = loader.load_scenarios(args.scen_dir, 
        [start_date, start_date], [scen for scen in range(args.nscens)])
    renew_scen_df = pd.concat([scen_dfs['Wind'], scen_dfs['Solar']], axis=1)
    
    renew_capacity = dict(zip(loader.gen_df['GEN UID'], loader.gen_df['PMax MW']))
    renew_gens = list(loader.template['ForecastRenewables'])
    timesteps = sorted(scen_dfs['Load'].index.get_level_values(1).unique())

    with open(Path(args.reserve_sim_dir,
            args.risk_type + '_risky_scens.p'), 'rb') as handle:
        risky_scens = pickle.load(handle)

    with open(Path(args.reserve_sim_dir,
            args.risk_type + '_alloc_scens.txt'), 'r') as f:
        alloc_scens = [int(scen.rstrip()) for scen in f]

    ## Get cost allocations
    allocation_dict = {}

    for scen in alloc_scens:
        allocs = pd.read_pickle(
            Path(args.cost_alloc_dir, 
            'allocation-' + str(scen) + '.p'))
        for gen in allocs:
            allocs[gen].index = allocs[gen].index.tz_localize('utc')
        allocation_dict[scen] = allocs

    renew_costs = {}

    if args.risk_type == 'daily':

        risk_score = {
            gen : reduce(
                lambda x, y: x.add(y, fill_value=0), 
                [allocation_dict[scen][gen] for scen in 
                alloc_scens]) / len(alloc_scens)
                for gen in renew_gens
        }
        
        for ts in timesteps:
            scen_df = renew_scen_df.loc[renew_scen_df.index.get_level_values(1)==ts]
            for gen in renew_gens:
                scens = scen_df[gen].values
                rscens = scen_df.droplevel(1).loc[alloc_scens, gen].values
                fcst = gen_data.loc[ts, ('fcst', gen)]
                pmax = renew_capacity[gen]
                risk = risk_score[gen].loc[ts].values[0]

                if risk > 0.:
                    prices, break_points, marginal_prices = get_renewable_cost_curve(
                        fcst, risk, scens, rscens, 0., pmax, 5)
                    renew_costs[(gen, ts)] = {'break_points': break_points, 
                                            'reliability_cost': marginal_prices,
                                            'slope': prices,
                                            'risk': risk,
                                            'fcst': fcst,
                                            'data': scens}
    elif args.risk_type == 'hourly':
    
        for h in range(24):
            
            risky_scens_hourly = list(risky_scens[h].keys())
            current_timestep = timesteps[h]

            risk_score = {
                gen : reduce(
                    lambda x, y: x.add(y, fill_value=0), 
                    [allocation_dict[scen][gen].loc[current_timestep] for scen in 
                    risky_scens_hourly]) / len(risky_scens[h]) 
                    for gen in renew_gens
            }
            
            scen_df = renew_scen_df.loc[
                renew_scen_df.index.get_level_values(1)==current_timestep]

            for gen in renew_gens:
                scens = scen_df[gen].values
                rscens = scen_df.droplevel(1).loc[risky_scens_hourly, gen].values
                fcst = gen_data.loc[current_timestep, ('fcst', gen)]
                pmax = renew_capacity[gen]
                risk = risk_score[gen].values[0]

                if risk > 0.:
                    prices, break_points, marginal_prices = get_renewable_cost_curve(
                        fcst, risk, scens, rscens, 0., pmax, 5)

                    renew_costs[(gen, current_timestep)] = {
                                            'break_points': break_points, 
                                            'reliability_cost': marginal_prices,
                                            'slope': prices,
                                            'risk': risk,
                                            'fcst': fcst,
                                            'data': scens}
    else:
        raise ValueError('unrecognized risk type')
    
    with open(Path(args.cost_curve_dir, 'cost-curve-' + 
            args.date + '.p'), 'wb') as handle:
        pickle.dump(renew_costs, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()