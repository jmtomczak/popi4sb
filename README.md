# POPI: Population-based Optimization for Parameter Identification
This repository provides code for the following paper:
- E. Weglarz-Tomczak, Jakub M. Tomczak, Agoston E. Eiben, Stanley Brul, "Population-based Optimization for Kinetic Parameters Identification in Glycolytic Pathway in *Saccharomyces cerevisiae*", 2020 (under review)

Here, we provide an implementation of five population-based optimizers (differential evolution (DE), (1+1)-evolutionary-strategy (ES), reversible differential evolution (RevDE), the univariate Gaussian estimationg of distribution algorithm (EDA), and RevDE + kNN and EDA + knn, where knn is the K-Nearest-Neighbor surrogate model) for parameter identification of dynamical models. The code is built on top of PySCeS (http://pysces.sourceforge.net/) and could be used for any dynamical model included in JWS Online database (https://jjj.bio.vu.nl/) or defined by a user. The model must be in the `.psc` format.

Here, we focus on the glycolysis in baker's yeast (*Saccharomyces cerevisiae*).

## Dependencies
- PySCeS (http://pysces.sourceforge.net/)
- Numpy (https://numpy.org/)
- SciPy (https://www.scipy.org/)
- Scikit-Learn (https://scikit-learn.org/)

## Running experiments
Experiments could by run by executing one of the files `run_experiment_X.py`, where `X` denotes an optimizer (DE, ES, EDA, EDAknn, RevDE, RevDEknn).

In the current code, there are two options (see paper for details):
- Case 1: The model without a mutation, `wolf1.psc`.
 - Case 2: The model with a mutation, `mutation1.psc`.
 
 ## Reference
 If you use this code in your research, please cite our paper:
```
 @article{glycolysis2020, 
  title={Population-based Optimization for Kinetic Parameters Identification in Glycolytic Pathway in Saccharomyces cerevisiae}, 
  author={W{\k{e}}glarz-Tomczak, Ewelina and Tomczak, Jakub M and Eiben, Agoston E and Brul, Stanley}, 
  journal={(under review)}, 
  year={2020}
}
```