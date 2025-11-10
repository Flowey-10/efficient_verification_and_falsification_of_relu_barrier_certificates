"""Define a dymamical system for an inverted pendulum"""

from typing import Tuple, Optional, List
from math import sqrt

import torch

from control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import Scenario, ScenarioList


class Linear4d(ControlAffineSystem):

    # Number of states and controls
    N_DIMS = 4
    N_CONTROLS = 1

    # State indices
    X = 0
    Y = 1
    Z = 2
    A = 3

    # Control indices
    U = 0

    def __init__(
        self,
        nominal_params: Scenario,
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
        scenarios: Optional[ScenarioList] = None,
        use_l1_norm: bool = False,
    ):
        """

        args:
            nominal_params: a dictionary giving the parameter values for the system.
            dt: the timestep to use for the simulation
            controller_dt: the timestep for the LQR discretization. Defaults to dt
            use_l1_norm: if True, use L1 norm for safety zones; otherwise, use L2
        raises:
            ValueError if nominal_params are not valid for this system
        """
        super().__init__(
            nominal_params,
            dt=dt,
            controller_dt=controller_dt,
            scenarios=scenarios,
            use_linearized_controller=False,
        )
        self.use_l1_norm = use_l1_norm
        self.P = torch.eye(self.n_dims).float()

    def validate_params(self, params: Scenario) -> bool:
        """Check if a given set of parameters is valid

        args:
            params: a dictionary giving the parameter values for the system.
        returns:
            True if parameters are valid, False otherwise
        """
        return True

    @property
    def n_dims(self) -> int:
        return Linear4d.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return []

    @property
    def n_controls(self) -> int:
        return Linear4d.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[Linear4d.X] = 2.0
        upper_limit[Linear4d.Y] = 2.0

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.tensor([0])
        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    def safe_mask(self, x):
        """Return the mask of x indicating safe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        # safe_mask = (x - torch.tensor([0.5, 1.5])).norm(dim=-1, p=1) < 0.5
        safe_mask = self.goal_mask(x)
        return safe_mask

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        unsafe_mask = torch.zeros_like(x[:, 0], dtype=torch.bool)

        h = x[:, Linear4d.X] + x[:, Linear4d.Y] ** 2

        unsafe_mask.logical_or_(h <= 0)

        return unsafe_mask

    def goal_mask(self, x):
        """Return the mask of x indicating points in the goal set

        args:
            x: a tensor of points in the state space
        """
        device = x.device
        goal_mask = (x - torch.tensor([0.5, 1.5]).to(device)).norm(dim=-1, p=1) < 0.5
        return goal_mask

    def _f(self, x: torch.Tensor, params: Scenario):
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            f: bs x self.n_dims x 1 tensor
        """
        # Extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, 1))
        f = f.type_as(x)

        # Extract the needed parameters
        x0_ = x[:, Linear4d.X]
        x1_ = x[:, Linear4d.Y]
        x2_ = x[:, Linear4d.Z]
        x3_ = x[:, Linear4d.A]

        f[:, Linear4d.X, 0] = -x0_
        f[:, Linear4d.Y, 0] = x0_ -2 * x1_
        f[:, Linear4d.Z, 0] = x0_ - 4 * x2_
        f[:, Linear4d.A, 0] = x0_ - 3 * x3_

        return f

    def _g(self, x: torch.Tensor, params: Scenario):
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            g: bs x self.n_dims x self.n_controls tensor
        """
        # Extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        g = torch.zeros((batch_size, self.n_dims, self.n_controls))
        g = g.type_as(x)

        return g

    def u_nominal(
        self, x: torch.Tensor, params: Optional[Scenario] = None
    ) -> torch.Tensor:
        return torch.zeros(x.shape[0], self.n_controls).to(x.device)
