# Physics Informed Neural Networks for Direct and Inverse Fluid Flow Problem
## MSc Thesis Project on UC3M
### Introduction 
This repository contains a Python/PyTorch implementation of a physics informed neural network (PINN) for solving fluid flow problems.
The code was developed as part of an MSc thesis project and includes all the necessary scripts and figures to reproduce the results. 
The project focuses on two main scenarios, each with its own corresponding folder containing different experiments. 
The main `.py` script is located in the root folder, and each experiment has its own specific folder where the results are automatically saved. 
The optimization process for each experiment's PINN is tracked using `TensorBoard` to monitor the loss function.
### One-dimensional fluid equations with slow area variation 
The code for each experiment in this section is relatively simple and only requires its own `.py` script.
Each script defines a PINN class tailored for the specific experiment, performs the training of the neural network, 
and calculates the corresponding results, which are then saved in their respective folders. 
There are four different experiments conducted:
+ Direct simulation
+ Inverse 1 parameter
+ Inverse 2 parameter
+ Inverse free distribution
### Two-dimensional Euler equations  
The code in this section is more complex and follows a well-organized structure. 
There are three scripts that define different classes used in this section: `Domain_classes.py` (contains classes associated with the domain,
distance function, and boundary extension), `PINN_forward_classes.py`, and `PINN_inverse_classes.py`` . 
Additionally, there are separate scripts for each experiment, where the PINNs are trained and the results (including plots) are obtained.
In each experiment, there are two flags: `RETRAIN` and `RETRAIN_DISTANCE`.
These flags can be used to avoid retraining the PINNs and distance/boundary neural networks by utilizing the previously saved parameters.

The experiments conducted in this section are as follows:
+ Experiment 1 : Comparison 1D vs 2D. Forward simulation of the 2D equations and comparison with the 1D model
+ Experiment 2 : Free-scale method. Testing the distance function on different domains 
+ Experiment 3 : Linear evolution method. Testing the boundary function on different domains
+ Experiment 4 : Inverse problem for AL.
+ Experiment 5 : Shock wave formation. (not totally finised, future work)
### Instructions 
To run the code in this repository, please follow these steps:
1. Clone the repository to your local machine.
2. Install the required dependencies.
3. Run the specific `.py` that correspond to the experiment you are interested in. You can choose either the parameters of the experiment and if 
you want to retrain the NN or not. 
5. Open the corresponding folder to see the results of the experiment.


Please note that further instructions or details for each experiment may be provided within their respective folders.

Feel free to explore the code and adapt it to your own needs. If you have any questions or encounter any issues,
please don't hesitate to contact the project author.

