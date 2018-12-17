# ml3

## File structure

### Euler

- <lib/euler_diffusion.f90> includes a module with a function for solving the diffusion equation using Forward Euler.
- <programs/euler.f90> reads `dx` from stdin, calls the function and writes the solution to file.

### Neural networks

- <programs/nn_simple.py> solves the diffusion equation and writes the solution to file, together with convergence.
- <programs/nn_params.py> reads parameters (activation function, optimiser and network architecture) from command line arguments, runs minimisation and writes cost and error to file.
- <programs/nn_params_analysis.py> reads the resulting consts and errors, sorts according to costs and creates a table.
- <Makefile.costs> runs <programs/nn_params.py> with a selection of parameters.
