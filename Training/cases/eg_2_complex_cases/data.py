import torch
import numpy as np
from casadi import project

import superp
import prob

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

############################################
# set default data type to double
############################################
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_dtype(torch.float32)
# torch.set_default_tensor_type(torch.FloatTensor)


############################################
# generating training data set
############################################

def gen_batch_data():
    def gen_full_data(region, len_sample):
        grid_sample = [torch.linspace(region[i][0], region[i][1], int(len_sample[i])) for i in range(prob.DIM)] # gridding each dimension
        mesh = torch.meshgrid(grid_sample) # mesh the gridding of each dimension
        flatten = [torch.flatten(mesh[i]) for i in range(len(mesh))] # flatten the list of meshes
        nn_input = torch.stack(flatten, 1) # stack the list of flattened meshes
        return nn_input



    full_init = gen_full_data(prob.INIT, superp.DATA_LEN_I)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')



    full_unsafe = gen_full_data(prob.UNSAFE, superp.DATA_LEN_U)
    full_unsafe = full_unsafe[:len(full_unsafe)]

    plt.show()

    full_domain = gen_full_data(prob.DOMAIN, superp.DATA_LEN_D)

    def batch_data(full_data, data_length, data_chunks, filter):
        l = list(data_length)
        batch_list = [torch.reshape(full_data, l + [prob.DIM])]
        for i in range(prob.DIM):
            batch_list = [tensor_block for curr_tensor in batch_list for tensor_block in list(curr_tensor.chunk(int(data_chunks[i]), i))]
        
        batch_list = [torch.reshape(curr_tensor, [-1, prob.DIM]) for curr_tensor in batch_list]
        batch_list = [curr_tensor[filter(curr_tensor)] for curr_tensor in batch_list]

        return batch_list

    def batch_data_unsafe(full_data, data_length, data_chunks, filter):
        l = list(data_length)
        batch_list = [torch.reshape(full_data, l + [prob.DIM])]
        for i in range(prob.DIM):
            batch_list = [tensor_block for curr_tensor in batch_list for tensor_block in
                          list(curr_tensor.chunk(int(data_chunks[i]), i))]

        batch_list = [torch.reshape(curr_tensor, [-1, prob.DIM]) for curr_tensor in batch_list]
        batch_list = [curr_tensor[filter(curr_tensor)] for curr_tensor in batch_list]

        return batch_list

    batch_init = batch_data(full_init, superp.DATA_LEN_I, superp.BLOCK_LEN_I, prob.cons_init)
    batch_unsafe = batch_data_unsafe(full_unsafe, superp.DATA_LEN_U, superp.BLOCK_LEN_U, prob.cons_unsafe)
    batch_domain = batch_data(full_domain, superp.DATA_LEN_D, superp.BLOCK_LEN_D, prob.cons_domain)

    return batch_init, batch_unsafe, batch_domain
