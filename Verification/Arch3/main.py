import numpy as np
import torch


import time
import torch.nn.functional as F
from Status import *
from SearchVerifier import *

import matplotlib.pyplot as plt
import itertools
from scipy.optimize import minimize, optimize
from scipy.optimize import LinearConstraint, NonlinearConstraint
import warnings
from visualization import *


from verification_of_relu.Darboux.darboux_1_20 import ann
from Quadratic_decay_Case import QuadraticDecay as QuadraticDecay_Case
from Quadratic_Decay import QuadraticDecay
from pydrake.solvers import * #SnoptSolver, ClarabelSolver

from Arch3_Case import Arch3 as Arch3_Case
from Arch3 import Arch3
from Status import NetworkStatus
from Function import RoAMatrix, LinearExp
from SearchVerifier import *
from intvalpy import lineqs

num_neuron = 32

model_name = "arch3_1_32.pt"


def get_dynamics_model(system_name):
    simulation_dt = 0.01
    controller_period = 0.01
    if system_name == "QuadraticDecay":
        nominal_params = {}
        scenarios = [
            nominal_params,
        ]

        # Define the dynamics model
        dynamics_model = QuadraticDecay(
            nominal_params,
            dt=simulation_dt,
            controller_dt=controller_period,
            scenarios=scenarios,
            use_l1_norm=False,
        )
    elif system_name == "arch3":
        nominal_params = {}
        scenarios = [
            nominal_params,
        ]

        # Define the dynamics model
        dynamics_model = Arch3(
            nominal_params,
            dt=simulation_dt,
            controller_dt=controller_period,
            scenarios=scenarios,
            use_l1_norm=False,
        )
    else:
        raise ValueError(f"Unknown system name: {system_name}")

    return dynamics_model


def get_case(system_name):
    if system_name == "QuadraticDecay":
        case = QuadraticDecay_Case()
    elif system_name == "arch3":
        case = Arch3_Case()
    else:
        raise ValueError(f"Unknown system name: {system_name}")
    return case


def main(system_name="arch3"):

    # Load Model
    print('Load Model')
    nnmodel = ann.gen_nn()



    nnmodel.load_state_dict(torch.load(model_name), strict=True)
    nnmodel = nnmodel.double()
    dynamics_model = get_dynamics_model(system_name)
    case = get_case(system_name)

    t_start = time.time()
    # safe_sample = dynamics_model.sample_safe(1)
    # unsafe_sample = dynamics_model.sample_unsafe(1)
    # while not all(dynamics_model.safe_mask(safe_sample)):
    #     safe_sample = dynamics_model.sample_safe(1)
    # while not all(dynamics_model.unsafe_mask(unsafe_sample)):
    #     unsafe_sample = dynamics_model.sample_unsafe(1)


    safe_sample = torch.tensor([[0.0, 0.0]])
    unsafe_sample = torch.tensor([[1.0, 1.0]])
    safept = safe_sample.unsqueeze(0)
    unsafept = unsafe_sample.unsqueeze(0)




    Search_prog = SearchVerifier(nnmodel, case)

    Search_prog.SV_CE(safept, unsafept) 



if __name__ == "__main__":
    main()
