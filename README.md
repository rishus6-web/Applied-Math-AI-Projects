# Solvers for Partial Differential Equations: A Comparative Study of Classical and AI-Driven Methods

## 1. Overview
This repository presents a comparative study and implementation of different numerical methods for solving time-dependent partial differential equations (PDEs), 
specifically the 2D Heat Equation. The project bridges the gap between classical, battle-tested techniques and modern, AI-driven paradigms.

The core of this work includes the following solvers:

   (1) A Classical Finite Difference Method (FDM) Solver, serving as a robust and verifiable baseline.
	
   (2) An AI-Driven Physics-Informed Neural Network (PINN), which learns the continuous solution in a mesh-free domain.
	
   (3) A Novel Spectral Physics-Informed Neural Network (SPINN), a custom-designed architecture that operates in the Fourier domain 

to more efficiently handle problems with significant high-frequency components.

This project showcases an end-to-end workflow, from theoretical formulation and implementation to validation and 
comparative analysis, demonstrating a deep understanding of both the underlying physics and the associated computational methodologies.


## 2. Visual Showcase: Simulation Results

The primary output of the solvers is a time-evolution animation of the temperature field. Below are the results from the classical FDM and the AI-driven PINN solver.

| **Finite Difference Method (FDM)** | **Physics-Informed Neural Network (PINN)** |
| :--------------------------------: | :----------------------------------------: |
| ![FDM Animation](FDM_heat_equation.gif) | ![PINN Animation](PINN_heat_equation.gif) |



## 3. The Physical Problem: 2D Heat Diffusion

   $$ \frac{\partial u}{\partial t} = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) $$  Where `u(t, x, y)` is the temperature and `α` is the thermal diffusivity. 


3. Classical Solver: Finite Difference Method

   
4. AI Solver: Physics-Informed Neural Network (PINN)


5. 
