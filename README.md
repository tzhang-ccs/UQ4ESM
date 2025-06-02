
# Auto-tuning framework for Earth System Models

## Introduction

This framework supports auto-tuning parameters in Earth System Models (ESMs). It covers the entire auto-tuning workflow, including parameter sampling, building machine learning surrogate models, configuring parameters within the model, submitting model simulations, calculating tuning performance metrics, and searching optimal parameters using optimization algorithms through iterative tuning.

The framework supports perturbed parameter ensemble (PPE) sampling using the Latin Hypercube Sampling (LHS) method and can be easily extended to other sampling strategies. It includes several optimization methods for parameter tuning, such as machine learning–based approaches, Bayesian Optimization (BO), Trust Region BO (TurBO), and derivative-free methods like the downhill simplex algorithm. The framework also allows for easy integration of custom optimization algorithms and user-defined performance metrics. 


## Required packages 

- numpy
- pandas
- xarray
- scipy
- gpytorch
- torch

## Key Files description

- para_set.json: Defines the tunable parameters and their corresponding value ranges.
- framework.py:  Implements the core functions of the auto-tuning workflow, including Latin Hypercube Sampling (with parallel sampling support), submitting simulations with modified parameter values, archiving model outputs, and managing the tuning process.
- metric.py: Computes the performance metric used to evaluate tuning results.

## Usage

The auto-tuning workflow consists of three key steps:

1. **Parameter Sampling**

2. **Defining the Tuning Performance Metric**

3. **Parameter Optimization Using Tuning Algorithms**

The following content provide a detailed explanation of each step.


### Parameter sampling
- set the tunable parameters and their ranges in the para_set.json file
```json
{
    "clubb_c8":[2.0,8.0],
    "zmconv_tau":[1800,14400.0],
    "nucleate_ice_subgrid":[1.0, 1.45],
    "zmconv_ke":[0.5E-6,10E-6],
    "p3_qc_accret_expon":[1,2],
    "p3_autocon_coeff":[1350, 30500.0],
    "p3_wbf_coeff":[0.1,1]
}
```
- generate the LHS sampling file
    - Initialize sampling and save the generated 20 samples to data/lhs_sampl.npy.
```python
from framework import uq

para_name = 'para_set.json'
foo = uq(para_name)
foo.lhs_sample(20,continue_run=False,continue_id=0)
```
    - If the initial sampling is incomplete, restart from a specific starting point.
```python
from framework import uq

para_name = 'para_set.json'
foo = uq(para_name)
foo.lhs_sample(10,continue_run=True,continue_id=10)
```

- generate the sampling simulation 
```python
foo.analyse('sample')
```
    - note: support parallel sampling by uncommenting the lines in the framework.py and set the number of threads
```python
            #pool = mp.Pool(1)
            #pool.starmap(self.run_case, [(i+1,d) for i,d in enumerate(self.sample_scaled)])
```

### Defining the Tuning Performance Metric
The ***metrics class*** in the metrics.py is response to calculate the tuning performace metric value. 

Users can design a custom tuning performance metric in the ***calc_metrics*** function.  Here, the example metric involves five physical quantities of interest (PQIs), including precipitation, shortwave cloud forcing, longwave cloud forcing, temperature at 850 hPa, and specific humidity at 850 hPa. The metric is defined by the following equations:

$$\sigma_{\mathrm{m}}^v=\sqrt{\sum_{i=1}^I w(i)\left(x_{\mathrm{m}}^v(i)-x_{\mathrm{o}}^v(i)\right)^2} $$
$$\sigma_{\mathrm{r}}^v=\sqrt{\sum_{i=1}^I w(i)\left(x_{\mathrm{r}}^v(i)-x_{\mathrm{o}}^v(i)\right)^2}$$
$$\chi=\frac{1}{N^v} \sum_{v=1}^{N^v}\frac{\sigma_{\mathrm{m}}^v}{\sigma_r^v}$$

where $x_m^v$ is the model output for one PQI, and $x_o^v$ is the corresponding observation. $x_r^v$ is the simulation using the default parameter values. $w$ denotes the weight due to the different grid area on a regular latitude-longitude grid on the sphere. $I$ represents the total number of grid points in model. $N^v$ indicates the number of PQIs chosen.

### Parameter Optimization Using Tuning Algorithms
- Tuning with TurBO method
```python
from framework import uq
para_name = 'para_set.json'
foo = uq(para_name)
foo.analyse(method='TurBO')
```

    - note:the parameters of TurBO method can be tuned in the ***framework.py***

```python
            turbo1 = Turbo1(
                f=f,  # Handle to objective function
                lb=f.lb,  # Numpy array specifying lower bounds
                ub=f.ub,  # Numpy array specifying upper bounds
                n_init=50,  # Number of initial bounds from an Latin hypercube design
                max_evals = 200,  # Maximum number of evaluations
                batch_size=10,  # How large batch size TuRBO uses
                verbose=True,  # Print information from each batch
                use_ard=True,  # Set to true if you want to use ARD for the GP kernel
                max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
                n_training_steps=60,  # Number of steps of ADAM to learn the hypers
                min_cuda=1024,  # Run on the CPU for small datasets
                device="cpu",  # "cpu" or "cuda"
                dtype="float64",  # float64 or float32
            )
```  

- Tuning with  Downhill Simplex method

```python
from framework import uq
para_name = 'para_set.json'
foo = uq(para_name)
foo.analyse(method='nelder-mead')
```

- note: The tuning process can be resumed by specifying the last simulation ID on line 218 of the ***framework.py*** file.

```python
self.tune_id = 68
```

- Tuning with BO method
```python
from framework import uq
para_name = 'para_set.json'
foo = uq(para_name)
foo.analyse(method='BO')
```


## how to cite
```

```
