import numpy as np
import torch

#from veri_util import *
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

from linear_4d_Case import Linear4d as Linear4d_Case
from linear_4d import Linear4d
from Status import NetworkStatus
from Function import RoAMatrix, LinearExp
from SearchVerifier import *
from intvalpy import lineqs


model_name = "linear4d_1_16.pt"



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
    elif system_name == "linear4d":
        nominal_params = {}
        scenarios = [
            nominal_params,
        ]

        # Define the dynamics model
        dynamics_model = Linear4d(
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
    if system_name == "linear4d":
        case = Linear4d_Case()
    else:
        raise ValueError(f"Unknown system name: {system_name}")
    return case


def main(system_name="linear4d"):
    # Load Model
    print('Load Model')
    nnmodel = ann.gen_nn()


    nnmodel.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')), strict=True)
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


    safe_sample = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    unsafe_sample = torch.tensor([[-2, -2, -2, -2]])
    safept = safe_sample.unsqueeze(0)
    unsafept = unsafe_sample.unsqueeze(0)


    Search_prog = SearchVerifier(nnmodel, case)

    Search_prog.SV_CE(safept, unsafept)



if __name__ == "__main__":
    main()

    a = 32
