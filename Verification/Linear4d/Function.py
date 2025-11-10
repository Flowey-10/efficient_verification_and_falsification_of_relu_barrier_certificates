import re
# from pydrake.solvers import MathematicalProgram, Solve
import numpy as np
import sys, os
import torch

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))
# from Modules.utils import *
# from Modules.NNet import NeuralNetwork as NNet
# from Scripts.Status import NeuronStatus, NetworkStatus
from pydrake.solvers import ClpSolver
from pydrake.solvers import MathematicalProgram


# Given a linear expression of a ReLU NN (Activation set $S$),
# return a set of linear constraints to formulate $\mathcal{X}(S)$

# Region of Activation (RoA) is the set of points that are activated by a ReLU NN
def RoAMatrix(model,
        S: dict = None, W_B: dict = None, r_B: dict = None, SSpace=[-2, 2]): #-> MathematicalProgram:
    # check if S is provided
    if S is None:
        # check if W_B, r_B are provided
        if W_B is None or r_B is None:
            raise ValueError("Activation set S or (W_B, r_B) are not provided")
    else:
        # if not, compute the linear expression of the output of the ReLU NN
        W_B, r_B, _, _ = LinearExp(model, S)
    final_layer_idx = len(W_B.keys())
    assert len(W_B) == len(r_B), "W_B and r_B must have the same amount of layers"
    # stack W_B and r_B constraints
    for keys, layer_info in W_B.items():
        if keys == 0:
            cons_W = layer_info
            cons_r = r_B[keys]
        elif keys == final_layer_idx:
            break
        else:
            cons_W = np.vstack([cons_W, layer_info]) 
            cons_r = np.hstack([cons_r, r_B[keys]])  


    return cons_W, cons_r


# Given a activation set $S$, return the linear expression of the output of the ReLU NN
def LinearExp(model, S: dict) -> (dict, dict, dict, dict):
    # Input: S: Activation set of a ReLU NN
    # Output: X: Linear expression of the output of the ReLU NN
    W_list = []
    r_list = []
    para_list = list(model.state_dict()) 
    i = 0
    while i < (len(para_list)):
        weight = model.state_dict()[para_list[i]]
        i += 1
        bias = model.state_dict()[para_list[i]]
        i += 1
        W_list.append(weight)
        r_list.append(bias)

    W_B = dict()
    r_B = dict()
    W_o = dict()
    r_o = dict()
    for keys, layer_info in S.items():
        # Get the current activation layer
        layer_act_list = torch.relu(torch.tensor(layer_info))
        layer_act_output_array = np.array(layer_act_list)
        layer_act_bound_array = np.array(layer_info)

        # compute output equivalent weight and bias for each layer
        W_o_layer = np.multiply(np.expand_dims(layer_act_output_array, -1), W_list[keys])
        r_o_layer = np.multiply(layer_act_output_array, r_list[keys])

        # compute boundary weight and bias for each layer
        W_B_layer = np.multiply(np.expand_dims(layer_act_bound_array, -1), W_list[keys])
        r_B_layer = np.multiply(layer_act_bound_array, r_list[keys])
        # add boundary condition to W_B and r_B
        if keys == 0:
            W_B[keys] = np.array(W_B_layer)
            r_B[keys] = np.array(r_B_layer)
            W_o[keys] = np.array(W_o_layer)
            r_o[keys] = np.array(r_o_layer)
        elif keys == len(S.keys()) - 1:
            W_o[keys] = np.array(np.matmul(W_list[keys], W_o[keys - 1]))
            r_o[keys] = np.array(np.matmul(W_list[keys], r_o[keys - 1]) + np.array(r_list[keys]))
        else:
            W_o[keys] = np.array(np.matmul(W_o_layer, W_o[keys - 1]))
            r_o[keys] = np.array(np.matmul(W_o_layer, r_o[keys - 1])) + np.array(r_o_layer)

            W_B[keys] = np.array(np.matmul(W_B_layer, W_o[keys - 1]))
            r_B[keys] = np.array(np.matmul(W_B_layer, r_o[keys - 1])) + np.array(r_B_layer)

    return W_B, r_B, W_o, r_o




def solver_lp(model, S, SSpace):
    # Input: X: Linear expression of the output of the ReLU NN
    # Output: X: Linear expression of the output of the ReLU NN
    # Create an empty MathematicalProgram named prog (with no decision variables,
    # constraints or costs)
    prog = MathematicalProgram()
    # Add two decision variables x[0], x[1].
    dim = next(model.children())[0].in_features
    x = prog.NewContinuousVariables(dim, "x")

    # Add linear constraints
    W_B, r_B, W_o, r_o = LinearExp(model, S)
    # prog = RoA(prog, x, model, S=S)
    prog = RoA(prog, x, model, S=None, W_B=W_B, r_B=r_B, SSpace=SSpace)
    # Output layer index
    index_o = len(S.keys()) - 1
    # Add linear constraints

    prog.AddLinearEqualityConstraint(np.array(W_o[index_o]), -np.array(r_o[index_o]), x)
    # tol = 1e-5
    # prog.AddLinearConstraint(np.array(W_o[index_o]), -np.array(r_o[index_o])-tol, -np.array(r_o[index_o])+tol, x)

    # Now solve the program.
    result = Solve(prog)
    # print(result.is_success())

    # print('check result:', np.matmul(W_o[index_o], result.GetSolution(x)) + r_o[index_o], W_o[index_o], r_o[index_o])
    # print('ref_result:', model.forward(torch.tensor(result.GetSolution(x)).float()))
    return result


if __name__ == "__main__":
    # architecture = [('relu', 2), ('relu', 32), ('relu', 32), ('linear', 1)]
    architecture = [('linear', 2), ('relu', 32), ('linear', 1)]

