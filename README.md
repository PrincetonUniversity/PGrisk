# PGrisk

PGrisk is a Python package for attributing the operational costs in power grid systems from economic dispatch to load demands and renewable generation units. 
It adapts the technique of Integrated Gradient in neural networks and quantifies the contribution of each input in the economic dispatch to the overall costs.
The underlying production cost modeling tools for running unit commitment and economic dispatch simulations are implemented in [Vatic](https://github.com/PrincetonUniversity/Vatic/tree/v0.4.1-a1).

# Getting started
### Requirements
+ Python 3.8 to 3.11
+ [Gurobi](https://www.gurobi.com/) mixed-integer linear programming solver

### Installation
First, obtain Vatic `v0.4.1-a1` by cloning the repository from command line:
```
git clone git@github.com:PrincetonUniversity/Vatic.git -b v0.4.1-a1 --single-branch
```
Next, download the testing grid [RTS-GMLC](https://github.com/GridMod/RTS-GMLC) and install Vatic by:
```
cd Vatic
git submodule init
git submodule update
pip install .
```
Finally, clone the repository and install PGrisk by running:
```
git clone git@github.com:PrincetonUniversity/PGrisk.git
cd PGrisk
pip install .
```

# Testing
PGrisk is packaged with an example in `test/RTS-GMLC_cost_attribution.ipynb` based on the RTS-GMLC grid to demonstrate its basic functionality.
