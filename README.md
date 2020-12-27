# POPI4SB: Population-based Parameter Identification for Dynamical Models of Biological Networks

This repository provides a Python framework for population-based parameter identification of dynamical models in systems biology. The code is built on top of PySCeS (http://pysces.sourceforge.net/) and could be used for any dynamical model included in JWS Online database (https://jjj.bio.vu.nl/) or defined by a user. The model must be in the `.psc` format.

## Dependencies
Before you download the code, please be sure to install the following packages:
- PySCeS (http://pysces.sourceforge.net/)
- Numpy (https://numpy.org/)
- SciPy (https://www.scipy.org/)
- Scikit-Learn (https://scikit-learn.org/)
 

## Use
1. Install all necessary packages.
2. Clone this repository.
3. Prepare/download your model (.psc).
4. Prepare a file (.json) with a specification of the model. If you use real data, please prepare the specification accordingly (i.e., the number of points, the beginning and the end of the experiment).
5. Prepare a file (.json) with a specification of an optimizer.
6. Update PySCeS solver information if necessary, or use the default setting.
7. Run `popi4sb.py` and follow the instructions.

## Features
- An integration with PySCeS.
- Easy-to-use to run simulators (i.e., dynamical models) in systems biology.
- Parameter identification of dynamical models using either one of the provided optimizers, or own optimizer.
- A possibility to add new optimizers (see `algorithms/population_optimization_algorithms.py`).
- An intuitive code structure.

 ## Reference
 If you use this code in your research, please cite our paper:
```
 @article{popi4sb, 
  title={Population-based Parameter Identification forDynamical Models of Biological Networks}, 
  author={Weglarz-Tomczak, Ewelina and Tomczak, Jakub M and Eiben, Agoston E and Brul, Stanley}, 
  journal={(under review)}, 
  year={2020}
}
```
