# PDAL-CPP

Primal dual augmented lagrangian - C++ version. 

## Build

### Build environemnt
The build environemnt is encapsulated in `.devcontainer/Dockerfile`. For VScode user, please install [VScode Remote Develpment](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack) and repoen the folder in container. For other users, please build `Dockerfile` manually.


### Build procedure
```
git clone  --recurse-submodules git@github.com:RoboticExplorationLab/PDAL-CPP.git
cd PDAL-CPP
mkdir build
cd build
cmake ..
make -j
```

## Reference

PDAL-CPP is nearly a line by line mapping of the corresponding [MATLAB version](https://github.com/RoboticExplorationLab/PDAL/blob/main/circularALPrimalDualCpp.m).

A short description of the notion and problem formulation can be found [here](https://github.com/RoboticExplorationLab/PDAL/blob/main/PDAL_notation.pdf).
