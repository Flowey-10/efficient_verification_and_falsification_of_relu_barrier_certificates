import sys, os

import torch

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))

from Status import *


class SearchInit:
    def __init__(self, model, case=None) -> None:
        self.model = model
        # self.dim = next(model.children())[0].in_features
        self.dim = model[0].weight.shape[1]  # first linear transformation
        self.case = case
        self.NStatus = NetworkStatus(model)

    def sample_in_region(self, region, len_sample):
        grid_sample = [torch.linspace(region[i][0], region[i][1], int(len_sample[i])) for i in range(len(region))]
        return torch.meshgrid(grid_sample)

    def connect_region(self, region, len_sample):
        mesh = self.sample_in_region(region, len_sample)
        flatten = [torch.flatten(mesh[i]) for i in range(len(mesh))]
        return torch.stack(flatten, 1)


    def get_zero_S(self, p_safe, p_unsafe, iter_lim=100):
        flag = True
        S = None
        epsilon_stop = 0.02 


        l_point = torch.tensor(p_safe)
        r_point = torch.tensor(p_unsafe)

        mid_point = (l_point + r_point) / 2
        while torch.norm(r_point - l_point) > epsilon_stop:
            mid_point = (l_point + r_point) / 2
            if torch.sign(self.model.forward(mid_point)) * torch.sign(self.model.forward(p_safe)) < 0:
                r_point = mid_point
            else:
                l_point = mid_point

        self.NStatus.get_netstatus_from_input_bound(l_point, r_point)
        S = self.NStatus.network_status_values

        return flag, S

    def initialization(self, input_safe: torch.tensor = None, input_unsafe: torch.tensor = None, m=1):
        # define default flag and S_init_Set
        flag, S_init_Set = False, {}
        # if we specify input_safe and input_unsafe, we use the input directly
        if input_safe is not None and input_unsafe is not None:
            m = input_safe.shape[0]
            safe_list_length = input_safe.shape[1]
            unsafe_list_length = input_unsafe.shape[1]
            x_safe = input_safe
            x_unsafe = input_unsafe
        elif input_safe is None:
            safe_list_length = len(self.safe_regions)
        elif input_unsafe is None:
            unsafe_list_length = len(self.unsafe_regions)
        else:
            safe_list_length = len(self.safe_regions)
            unsafe_list_length = len(self.unsafe_regions)
        # iterate through all safe and unsafe regions to find all intial activation sets
        for i in range(safe_list_length):
            for j in range(unsafe_list_length):
                if input_safe is None and input_unsafe is None:
                    x_safe = self.sample_in_region(self.safe_regions[i], [m, 1])
                    x_unsafe = self.sample_in_region(self.unsafe_regions[j], [m, 1])
                else:
                    x_safe = input_safe[i]
                    x_unsafe = input_unsafe[j]
                if not flag:
                    for k in range(m):
                        flag, S = self.get_zero_S(x_safe[0], x_unsafe[0])  # only check the first unsafe point
                        if flag:
                            S_init_Set[j] = S
        if not flag:
            raise ("Initialization failed")
        return S_init_Set


if __name__ == "__main__":
    architecture = [('linear', 2), ('relu', 32), ('linear', 1)]
    model = NNet(architecture)
    trained_state_dict = torch.load("./Phase1_Scalability/darboux_1_32.pt")
    trained_state_dict = {f"layers.{key}": value for key, value in trained_state_dict.items()}
    model.load_state_dict(trained_state_dict, strict=True)
    # case = PARA.CASES[0]
    Search = SearchInit(model)
    # (0.5, 1.5), (0, -1)
    S_init_Set = Search.initialization(torch.tensor([[[0.5, 1.7]]]), torch.tensor([[[-1, -1.6]]]))
    # S_init_Set = Search.initialization(input_safe=torch.tensor([[[0.5, 1.5]]]), input_unsafe=torch.tensor([[[-1, 0]]]))
    print(S_init_Set)

