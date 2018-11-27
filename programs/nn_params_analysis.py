import numpy as np

activation_functions = []
optimisers = []
architectures = []
costs = []
errors = []

with open("data/nn_costs.dat", "r") as infile:
    for line in infile:
        if not "nan" in line:
            words = line.split()
            activation_functions.append(words[0])
            optimisers.append(words[1])
            architectures.append(words[2])
            costs.append(float(words[3]))
            errors.append(float(words[4]))

sorting_indices = np.argsort(costs)

activation_functions = np.asarray(activation_functions)[sorting_indices]
optimisers = np.asarray(optimisers)[sorting_indices]
architectures = np.asarray(architectures)[sorting_indices]
costs = np.asarray(costs)[sorting_indices]
errors = np.asarray(errors)[sorting_indices]

with open("data/nn_cost_table_small.dat", "w") as outfile:
    outfile.write("{Activation function} Optimiser Architecture Cost Error\n")
    for i in range(17):
        opt = optimisers[i]
        if opt != "Adam":
            alpha, gamma = opt.split(",")
            opt = (
                r"$\alpha=\num[scientific-notation=true]{%s},\gamma=\num[scientific-notation=true]{%s}$"
                % (alpha, gamma)
            )
        outfile.write(
            "%s %s %s %g %g\n"
            % (activation_functions[i], opt, architectures[i], costs[i], errors[i])
        )

with open("data/nn_cost_table.dat", "w") as outfile:
    outfile.write("{Activation function} Optimiser Architecture Cost Error\n")
    for i in range(len(costs)):
        opt = optimisers[i]
        if opt != "Adam":
            alpha, gamma = opt.split(",")
            opt = r"$\alpha=\num{%s},\gamma=\num{%s}$" % (alpha, gamma)
        outfile.write(
            "%s %s %s %g %g\n"
            % (activation_functions[i], opt, architectures[i], costs[i], errors[i])
        )
