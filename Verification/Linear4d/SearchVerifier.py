# from Verifier.Verification import *
from collections import deque
from Function import *
from verification_of_relu_submit.Linear4d.Search import Search
import time
import matplotlib.pyplot as plt


class SearchVerifier(Search):
    def __init__(self, model, case) -> None:
        super().__init__(model, case)


    def SV_CE(self, spt, uspt):
        result_info = {}


        t_start = time.time()
        self.Specify_point(spt, uspt) 
        self.Enumerate_Init_Activation()
        t_enumerate_end = time.time()
        print(f'enumeration costs', t_enumerate_end - t_start, 'seconds')
        self.facet_Enumeration(self.activation_init[0])
        t_facet_enumerate = time.time()
        print(f'facet enumeration costs', t_facet_enumerate - t_enumerate_end, 'seconds')
        flag = self.verification()
        if flag:
            print("Verification Success!")
        else:
            print("Verification failed!")
        t_end = time.time()
        print(f'The verification costs', t_end - t_facet_enumerate, 'seconds')
        print(f'The experiment spends', t_end - t_start, 'seconds')



if __name__ == "__main__":
    # CBF Verification
    l = 1
    n = 128
    case = ObsAvoid()
    hdlayers = []
    for layer in range(l):
        hdlayers.append(('relu', n))
    architecture = [('linear', 3)] + hdlayers + [('linear', 1)]
    model = NNet(architecture)
    trained_state_dict = torch.load(f"Phase1_Scalability/models/obs_{l}_{n}.pt")
    trained_state_dict = {f"layers.{key}": value for key, value in trained_state_dict.items()}
    model.load_state_dict(trained_state_dict, strict=True)

    spt = torch.tensor([[[-1.0, 0.0, 0.0]]])
    uspt = torch.tensor([[[0.0, 0.0, 0.0]]])
    # Search Verification and output Counter Example
    Search_prog = SearchVerifier(model, case)
    veri_flag, ce, info = Search_prog.SV_CE(spt, uspt)
    if veri_flag:
        print('Verification successful!')
        print("Search info:", info)
    else:
        print('Verification failed!')
        print('Counter example:', ce)

