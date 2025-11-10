import sys, os
import copy
from reprlib import recursive_repr

import numpy as np
import torch
from numpy import ubyte

from visualization import *

from dreal import *
import pickle
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))
from itertools import combinations
from collections import deque

from pydrake.solvers import ClpSolver, ChooseBestSolver, GurobiSolver
from pydrake.solvers import *
from SearchInit import SearchInit

from pydrake.solvers import Solve
from pydrake.solvers import MathematicalProgram
from Status import *
from Function import RoAMatrix, LinearExp
from copy import deepcopy
from gilp.simplex import LP
from gilp.visualize import simplex_visual
from intvalpy import lineqs
import matplotlib.pyplot as plt




class Search(SearchInit):
    def __init__(self, model, case=None) -> None:
        super().__init__(model, case)
        self.model = model
        self.case = case
        self.NStatus = NetworkStatus(model)
        self.verbose = False
        self.activation_init = [] 
        self.acti_bound_propa = []
        self.veri_acti = []
        self.count_num = 0

    def Specify_point(self, safe_point: torch.tensor, unsafe_point: torch.tensor):
        S_init = self.initialization(input_safe=safe_point, input_unsafe=unsafe_point)
        self.S_init = S_init 

    def Enumerate_Init_Activation(self):
        for activationset in self.S_init.values():
            self.recursive_enumeration(activationset, 0, 0, 0)

    def isRealActivation(self, S):
        for layer_idx, (_, layer_status) in enumerate(S.items()):
            if layer_idx != len(S) - 1:
                if not torch.tensor(layer_status).all():
                    return False

        return True
    def recursive_enumeration(self, activation_set, ind_layer, ind_neuron, count_num):
        if self.isRealActivation(activation_set):
            if self.TestOne(activation_set):
                self.activation_init.append(deepcopy(activation_set))
                return True
        for layer_idx, (_, layer_status) in enumerate(activation_set.items()):
            if layer_idx != len(activation_set) - 1 and layer_idx >= ind_layer:
                for neuron_idx, neuron_status in enumerate(layer_status):
                    if neuron_status == 0:
                        flag = False
                        if layer_idx > ind_layer:
                            activation_set[layer_idx][neuron_idx] = 1
                            flag = self.recursive_enumeration(activation_set, layer_idx, 0, count_num + 1)
                            if flag:
                                return True
                            activation_set[layer_idx][neuron_idx] = -1
                            flag = self.recursive_enumeration(activation_set, layer_idx, 0, count_num + 1)
                        elif layer_idx == ind_layer:
                            if neuron_idx >= ind_neuron:
                                activation_set[layer_idx][neuron_idx] = 1
                                flag = self.recursive_enumeration(activation_set, layer_idx, neuron_idx, count_num + 1)
                                if flag:
                                    return True
                                activation_set[layer_idx][neuron_idx] = -1
                                flag = self.recursive_enumeration(activation_set, layer_idx, neuron_idx, count_num + 1)
                        if flag:
                            return True
    def TestValid(self):
        for item in self.activation_init:
            flag = self.TestOne(item)
            if flag:
                print("successful TestValid!")
                return item


    def TestOne(self, S):
        flag = False
        count_equ = 0 
        SSpace = [-2, 2] 

        W_B, r_B, W_o, r_o = LinearExp(self.model, S)
        cons_W, cons_r = RoAMatrix(self.model, S=None, W_B=W_B, r_B=r_B, SSpace=self.case.SSpace)
        bdyW = W_o[len(W_o) - 1]
        bdyr = r_o[len(r_o) - 1]
        # feasibility test
        solver = NloptSolver()
        for i in range(cons_W.shape[0]):
            prog1 = MathematicalProgram()
            x = prog1.NewContinuousVariables(self.dim, "x")
            linear_constraint = prog1.AddLinearConstraint(A=cons_W, lb=-cons_r,
                                                          ub=np.inf * np.ones(len(cons_r)), vars=x)
            prog1.AddCost(np.dot(cons_W[i], x))
            prog1.AddLinearEqualityConstraint(Aeq=bdyW, beq=-bdyr, vars=x)
            prog2 = MathematicalProgram()
            y = prog2.NewContinuousVariables(self.dim, "y")
            linear_constraint = prog2.AddLinearConstraint(A=cons_W, lb=-cons_r,
                                                          ub=np.inf * np.ones(len(cons_r)), vars=y)
            prog2.AddCost(np.dot(-cons_W[i], y))
            prog2.AddLinearEqualityConstraint(bdyW, -bdyr, y)
            result1 = solver.Solve(prog1)
            result2 = solver.Solve(prog2)
            if result1.is_success() and result2.is_success():
                if result1.get_optimal_cost() == -result2.get_optimal_cost():
                    if count_equ == 0:
                        W_equ = cons_W[i]
                    else:
                        W_equ = np.vstack([W_equ, cons_W[i]])
                    count_equ = count_equ + 1

        # solve for the equality constraint
        prog1 = MathematicalProgram()
        x = prog1.NewContinuousVariables(self.dim, "x")
        linear_constraint = prog1.AddLinearConstraint(A=cons_W, lb=-cons_r,
                                                      ub=np.inf * np.ones(len(cons_r)), vars=x)
        prog1.AddCost(np.dot(bdyW[0], x))
        prog1.AddLinearEqualityConstraint(bdyW, -bdyr, x)
        prog2 = MathematicalProgram()
        y = prog2.NewContinuousVariables(self.dim, "y")
        linear_constraint = prog2.AddLinearConstraint(A=cons_W, lb=-cons_r,
                                                      ub=np.inf * np.ones(len(cons_r)), vars=y)
        prog2.AddCost(np.dot(-bdyW[0], y))
        prog2.AddLinearEqualityConstraint(bdyW, -bdyr, y)
        result1 = solver.Solve(prog1)
        result2 = solver.Solve(prog2)
        if result1.is_success() and result2.is_success():
            if count_equ == 0:
                W_equ = bdyW
                W_equ = np.vstack([W_equ, -bdyW[0]])
            else:
                W_equ = np.vstack([W_equ, bdyW[0]])
                W_equ = np.vstack([W_equ, -bdyW[0]])
            count_equ = count_equ + 1
        try:
            rank_equ = np.linalg.matrix_rank(W_equ)
        except:
            rank_equ = 0
            flag = False  # this means W_equ has no implicit inequality
        if rank_equ == 1:
            flag = True
        return flag

    def facet_Enumeration(self, S):
        print("facet_enumeration_start!")
        initial_acti = []
        self.initial_acti = initial_acti
        visited_acti = []
        self.visited_set = []
        self.visited_acti = visited_acti
        SSpace = [-2, 2]
        initial_acti.append(S)
        solver = NloptSolver()
        while initial_acti != []:
            item = initial_acti.pop(0)
            visited_acti.append(item)
            W_B, r_B, W_o, r_o = LinearExp(self.model, item)
            cons_W, cons_r = RoAMatrix(self.model, S=None, W_B=W_B, r_B=r_B, SSpace=self.case.SSpace)
            bdyW = W_o[len(W_o) - 1]
            bdyr = r_o[len(r_o) - 1]
            for i in range(cons_W.shape[0]):
                prog = MathematicalProgram()
                x = prog.NewContinuousVariables(self.dim, "x")
                linear_constraint = prog.AddLinearConstraint(A=cons_W, lb=-cons_r,
                                                              ub=np.inf * np.ones(len(cons_r)), vars=x)
                prog.AddLinearEqualityConstraint(cons_W[i], -cons_r[i], x)
                prog.AddLinearEqualityConstraint(bdyW, -bdyr, x)
                result = solver.Solve(prog)
                x_valid = None
                if result.is_success():
                    x_valid = result.GetSolution()
                    initial_acti = self.point_Enumerate_Acti(torch.tensor(x_valid), initial_acti)


            initial_acti = [x for x in initial_acti if x not in visited_acti]
        self.veri_acti = visited_acti


    def point_Enumerate_Acti(self, pt, temp_acti_set):
        new_enumerate = []
        self.NStatus.get_netstatus_from_input(pt)
        if self.NStatus.network_status_values not in self.visited_set:
            self.recursive_enumeration_valid(self.NStatus.network_status_values, new_enumerate, 0, 0)
            self.visited_set.append(deepcopy(self.NStatus.network_status_values))
        return temp_acti_set + new_enumerate

    def recursive_enumeration_valid(self, activation_set, storelist, ind_layer, ind_neuron):
        if self.isRealActivation(activation_set):
            if activation_set not in self.initial_acti and activation_set not in self.visited_acti and activation_set not in storelist:
                if self.TestOne(activation_set):
                    storelist.append(deepcopy(activation_set))

            return
        flag = False # judge whether a stable neruon is enumerated
        for layer_idx, (_, layer_status) in enumerate(activation_set.items()):
            if flag:
                break
            if layer_idx != len(activation_set) - 1 and layer_idx >= ind_layer:
                if flag:
                    break
                for neuron_idx, neuron_status in enumerate(layer_status):
                    if flag:
                        break
                    if neuron_status == 0:
                        if layer_idx > ind_layer:
                            activation_set[layer_idx][neuron_idx] = 1
                            self.recursive_enumeration_valid(deepcopy(activation_set), storelist, layer_idx, 0)
                            activation_set[layer_idx][neuron_idx] = -1
                            self.recursive_enumeration_valid(deepcopy(activation_set), storelist, layer_idx, 0)
                            flag = True
                        elif layer_idx == ind_layer:
                            if neuron_idx >= ind_neuron:
                                activation_set[layer_idx][neuron_idx] = 1
                                self.recursive_enumeration_valid(deepcopy(activation_set), storelist, layer_idx, neuron_idx)
                                activation_set[layer_idx][neuron_idx] = -1
                                self.recursive_enumeration_valid(deepcopy(activation_set), storelist, layer_idx, neuron_idx)
                                flag = True



    def plot_neural(self):
        gridsize = 400
        nx = torch.linspace(-1, 1, gridsize)
        ny = torch.linspace(-1, 1, gridsize)
        vx, vy = torch.meshgrid(nx, ny)
        data = np.dstack([vx.reshape([gridsize, gridsize, 1]), vy.reshape([gridsize, gridsize,1])])
        data = torch.Tensor(data.reshape(gridsize * gridsize, 2))
        z = self.model.forward(data).reshape((gridsize, gridsize))
        plt.contour(vx.detach().numpy(), vy.detach().numpy(), z.detach().numpy(), [0], colors='r') # cmap='viridis'

    def plot_vector(self):
        gridsize = 10
        nx = np.linspace(-1, 1, gridsize)
        ny = np.linspace(-1, 1, gridsize)
        vx, vy = np.meshgrid(nx, ny)
        # dx = vy + 2 * vx * vy
        # dy = -vx + 2 * vx ** 2 - vy ** 2
        dx = vx - vx**3 + vy - vx * vy ** 2
        dy = -vx + vy - vx ** 2 * vy - vy ** 3
        # dx = -vy
        # dy = vx - 3 * (1 - vx ** 2) * vy
        # x0_dot = -x[1]
        # x1_dot = x[0] - (1 - x[0] ** 2) * x[1]
        plt.streamplot(vx, vy, dx, dy)
        # plt.quiver(vx, vy, dx, dy, 1)


    def verification(self):
        bbounds = np.array([[-5, 5], [-5, 5]])
        flag_positive = True

        #  remove the duplicated
        new_list = []
        for i in self.veri_acti:
            if i not in new_list:
                new_list.append(i)
        self.veri_acti = new_list
        print(f'There are', len(self.veri_acti), 'to be verified')
        errorcnt = 0
        solver_status_cnt = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        solver_status_wtcost = [0, 0, 0, 0, 0, 0, 0, 0]
        Solver = NloptSolver()
        colorlist = ['g', 'c', 'm', 'y', 'b', 'g', 'c', 'm', 'y', 'b'] * 1000
        cnt = 0
        for ind, item in enumerate(self.veri_acti):
            W_B, r_B, W_o, r_o = LinearExp(self.model, item)
            cons_W, cons_r = RoAMatrix(self.model, S=None, W_B=W_B, r_B=r_B, SSpace=self.case.SSpace)
            prog = MathematicalProgram()
            x = prog.NewContinuousVariables(self.dim, "x")
            ll = len(W_o) - 1
            SSpace = [-2, 2]
            consequ = prog.AddLinearEqualityConstraint(W_o[ll], -r_o[ll], x)
            prog.AddLinearConstraint(A=cons_W, lb=-cons_r,
                                     ub=np.inf * np.ones(len(cons_r)), vars=x)
            result = Solver.Solve(prog)
            solver_status_wtcost[result.get_solver_details().status] += 1

            prog.AddCost(-np.dot(W_o[ll], self.case.f_x(x))[0][0])

            result = Solver.Solve(prog)
            cnt = cnt + 1
            solver_status_cnt[result.get_solver_details().status] += 1
            if result.get_solver_details().status == 5:
                print(result.get_optimal_cost())
                print(result.GetSolution())
                print(result.is_success())
            if result.is_success() == False:
                print("oh, it's wrong!")
                print(result.get_optimal_cost())
                flag_positive = False
                break
            else:
                if result.get_optimal_cost() < 0:
                    print("optimal cost is wrong!")
                    print(result.get_optimal_cost())
                    flag_positive = False
                    break
        print(errorcnt)
        print(solver_status_cnt)
        print(solver_status_wtcost)

        # examination for whether initial states are in the barrier certificates
        included_flag = True
        for ind, item in enumerate(self.veri_acti):
            prog = MathematicalProgram()
            x = prog.NewContinuousVariables(self.dim, "x")
            W_B, r_B, W_o, r_o = LinearExp(self.model, item)
            cons_W, cons_r = RoAMatrix(self.model, S=None, W_B=W_B, r_B=r_B, SSpace=self.case.SSpace)
            ll = len(W_o) - 1
            consequ = prog.AddLinearEqualityConstraint(W_o[ll], -r_o[ll], x)
            prog.AddLinearConstraint(A=cons_W, lb=-cons_r,
                                     ub=np.inf * np.ones(len(cons_r)), vars=x)
            prog.AddCost(np.sum(x ** 2) - 0.04)
            result = Solver.Solve(prog)
            if result.get_optimal_cost() <= 0:
                included_flag = False
                break  # if failed for one part, it is all failed

        if included_flag:
            print("the invariant set is in safe set-optimization verification")
        else:
            print("the invariant set is not in safe set-optimization verification")

        if included_flag == True:
            # test using SMT solver
            x = Variable("x", Variable.Real)
            y = Variable("y", Variable.Real)
            z = Variable("z", Variable.Real)
            a = Variable("a", Variable.Real)
            variables = np.array([x, y, z, a])

            for ind, item in enumerate(self.veri_acti):
                f_sat_part = 1
                W_B, r_B, W_o, r_o = LinearExp(self.model, item)
                cons_W, cons_r = RoAMatrix(self.model, S=None, W_B=W_B, r_B=r_B, SSpace=self.case.SSpace)
                ll = len(W_o) - 1
                f_sat_part = 0.04 - x ** 2 - y ** 2 - z ** 2 - a ** 2 >= 0
                for i in range(len(variables)):
                    f_sat_part = And(f_sat_part, variables[i] >= -1e8)
                    f_sat_part = And(f_sat_part, variables[i] <= 1e8)
                region_constraint = np.dot(cons_W, variables)
                for i in range(cons_W.shape[0]):
                    f_sat_part = And(f_sat_part, region_constraint[i] >= -cons_r[i])

                f_sat_part = And(f_sat_part, np.dot(W_o[ll], variables)[0] + r_o[ll][0] == 0)
                result = CheckSatisfiability(f_sat_part, 0.001)
                if result != None:
                    included_flag = False
                    break

        if included_flag:
            print("the invariant set is in safe set-SMT verification")
        else:
            print("the invariant set is not in safe set-SMT verification")

        # verification of whether the unsafe set intersects the invariant set
        included_flag = True
        for ind, item in enumerate(self.veri_acti):
            prog = MathematicalProgram()
            x = prog.NewContinuousVariables(self.dim, "x")
            W_B, r_B, W_o, r_o = LinearExp(self.model, item)
            cons_W, cons_r = RoAMatrix(self.model, S=None, W_B=W_B, r_B=r_B, SSpace=self.case.SSpace)
            ll = len(W_o) - 1
            consequ = prog.AddLinearEqualityConstraint(W_o[ll], -r_o[ll], x)
            prog.AddLinearConstraint(A=cons_W, lb=-cons_r,
                                     ub=np.inf * np.ones(len(cons_r)), vars=x)
            # prog.AddCost(-np.dot(W_o[ll], self.case.f_x(x))[0][0])
            prog.AddCost(np.sum((x - 3) ** 2) - 1)
            result = Solver.Solve(prog)
            if result.get_optimal_cost() <= 0:
                # print(result.GetSolution())
                included_flag = False
                break  # if failed for one part, it is all failed
        if included_flag:
            print("the invariant set does not intersect unsafe set-optimization verification")
        else:
            print("the invariant set does intersect unsafe set-optimization verification")

        # Test using SMT solver
        if included_flag:
            x = Variable("x", Variable.Real)
            y = Variable("y", Variable.Real)
            z = Variable("z", Variable.Real)
            a = Variable("a", Variable.Real)
            variables = np.array([x, y, z, a])
            f_sat = 1
            for ind, item in enumerate(self.veri_acti):
                f_sat_part = 1
                W_B, r_B, W_o, r_o = LinearExp(self.model, item)
                cons_W, cons_r = RoAMatrix(self.model, S=None, W_B=W_B, r_B=r_B, SSpace=self.case.SSpace)
                ll = len(W_o) - 1
                f_sat_part = (x - 3) ** 2 + (y - 3) ** 2 + (z - 3) ** 2 + (a - 3) ** 2 - 1 <= 0
                region_constraint = np.dot(cons_W, variables)
                for i in range(len(variables)):
                    f_sat_part = And(f_sat_part, variables[i] >= -1e8)
                    f_sat_part = And(f_sat_part, variables[i] <= 1e8)
                for i in range(cons_W.shape[0]):
                    f_sat_part = And(f_sat_part, region_constraint[i] >= -cons_r[i])

                f_sat_part = And(f_sat_part, np.dot(W_o[ll], variables)[0] + r_o[ll][0] == 0)
                result = CheckSatisfiability(f_sat_part, 0.001)
                if result != None:
                    included_flag = False
                    break
        if included_flag:
            print("the invariant set does not intersect unsafe set-SMT verification")
        else:
            print("the invariant set does intersect unsafe set-SMT verification")

        if flag_positive and included_flag:
            print("verification success_real")
        else:
            print("Verification failed_real")
            return False
        return True

    def vectorfield_dreal(self, x):
        x0_dot = -x[0]
        x1_dot = x[0] - 2 * x[1]
        x2_dot = x[0] - 4 * x[2]
        x3_dot = x[0] - 3 * x[3]
        return np.array([x0_dot, x1_dot, x2_dot, x3_dot])

    def TestOneTemp(self, S, fig, ax):
        eps1 = 1e-8
        flag = False
        count_equ = 0 
        SSpace = [-2, 2] 

        W_B, r_B, W_o, r_o = LinearExp(self.model, S)
        cons_W, cons_r = RoAMatrix(self.model, S=None, W_B=W_B, r_B=r_B, SSpace=self.case.SSpace)
        bdyW = W_o[len(W_o) - 1]
        bdyr = r_o[len(r_o) - 1]


        solver = ClpSolver()
        for i in range(cons_W.shape[0]):
            prog1 = MathematicalProgram()
            x = prog1.NewContinuousVariables(self.dim, "x")
            linear_constraint = prog1.AddLinearConstraint(A=cons_W, lb=-cons_r,
                                                          ub=np.inf * np.ones(len(cons_r)), vars=x)
            prog1.AddCost(np.dot(cons_W[i], x))
            prog1.AddLinearEqualityConstraint(Aeq=bdyW, beq=-bdyr, vars=x)
            prog2 = MathematicalProgram()
            y = prog2.NewContinuousVariables(self.dim, "y")
            linear_constraint = prog2.AddLinearConstraint(A=cons_W, lb=-cons_r,
                                                          ub=np.inf * np.ones(len(cons_r)), vars=y)
            prog2.AddCost(np.dot(-cons_W[i], y))
            prog2.AddLinearEqualityConstraint(bdyW, -bdyr, y)
            result1 = solver.Solve(prog1)
            result2 = solver.Solve(prog2)
            if result1.is_success() and result2.is_success():
                if abs(result1.get_optimal_cost() + result2.get_optimal_cost()) < eps1: 
                    if count_equ == 0:
                        W_equ = cons_W[i]
                    else:
                        W_equ = np.vstack([W_equ, cons_W[i]])
                    count_equ = count_equ + 1
        # solve for the equality constraint
        prog1 = MathematicalProgram()
        x = prog1.NewContinuousVariables(self.dim, "x")
        linear_constraint = prog1.AddLinearConstraint(A=cons_W, lb=-cons_r,
                                                      ub=np.inf * np.ones(len(cons_r)), vars=x)
        prog1.AddCost(np.dot(bdyW[0], x))
        prog1.AddLinearEqualityConstraint(bdyW, -bdyr, x)
        prog2 = MathematicalProgram()
        y = prog2.NewContinuousVariables(self.dim, "y")
        linear_constraint = prog2.AddLinearConstraint(A=cons_W, lb=-cons_r,
                                                      ub=np.inf * np.ones(len(cons_r)), vars=y)
        prog2.AddCost(np.dot(-bdyW[0], y))
        prog2.AddLinearEqualityConstraint(bdyW, -bdyr, y)
        result1 = solver.Solve(prog1)
        result2 = solver.Solve(prog2)
        if result1.is_success() and result2.is_success():
            if abs(result1.get_optimal_cost() + result2.get_optimal_cost()) < eps1: 
                if count_equ == 0:
                    W_equ = bdyW
                    W_equ = np.vstack([W_equ, -bdyW[0]])
                else:
                    W_equ = np.vstack([W_equ, bdyW[0]])
                    W_equ = np.vstack([W_equ, -bdyW[0]])
                count_equ = count_equ + 1
        try:
            rank_equ = np.linalg.matrix_rank(W_equ)
        except:
            rank_equ = 0
            flag = False  # this means W_equ has no implicit inequality
        if rank_equ == 1:
            flag = True
        plt.show()
        return flag



if __name__ == "__main__":
    architecture = [('linear', 2), ('relu', 64), ('linear', 1)]
    model = NNet(architecture)
    trained_state_dict = torch.load("./Phase1_Scalability/models/darboux_1_64.pt")
    trained_state_dict = {f"layers.{key}": value for key, value in trained_state_dict.items()}
    model.load_state_dict(trained_state_dict, strict=True)
    # case = PARA.CASES[0]
    Search = Search(model)
    # (0.5, 1.5), (0, -1)
    Search.Specify_point(torch.tensor([[[0.5, 1.5]]]), torch.tensor([[[-1, 0]]]))
    # print(Search.S_init)

    # Search.Filter_S_neighbour(Search.S_init[0])
    # Possible_S = Search.Possible_S(Search.S_init[0], Search.Filter_S_neighbour(Search.S_init[0]))
    # print(Search.Filter_S_neighbour(Search.S_init[0]))
    unstable_neurons_set, pair_wise_hinge = Search.BFS(Search.S_init[0])
    # unstable_neurons_set = Search.BFS(Possible_S)
    # print(unstable_neurons_set)
    print(len(unstable_neurons_set))
    print(len(pair_wise_hinge))

    ho_hinge = Search.hinge_search(unstable_neurons_set, pair_wise_hinge)
    print(len(ho_hinge))

