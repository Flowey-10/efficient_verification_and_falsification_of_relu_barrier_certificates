import torch
import numpy as np
import superp
import prob

import matplotlib.pyplot as plt

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

    # def data_filter(region, data):
    #     mask = torch.ones(data.shape[0], dtype=torch.bool)
    #     for dim in range(data.shape[1]):
    #         mask_dim = (data[:, dim] < region[dim][0]) | (data[:, dim] > region[dim][1])
    #         mask = mask & mask_dim
    #     return data[mask]

    full_init = gen_full_data(prob.INIT, superp.DATA_LEN_I)
    plt.plot(full_init[:, 0], full_init[:, 1])

    full_unsafe1 = gen_full_data(prob.UNSAFE_1, superp.DATA_LEN_U / 2)
    full_unsafe1 = full_unsafe1[:(len(full_unsafe1))]
    plt.plot(full_unsafe1[:, 0], full_unsafe1[:, 1])
    full_unsafe2 = gen_full_data(prob.UNSAFE_2, superp.DATA_LEN_U / 2)
    full_unsafe2 = full_unsafe2[:len(full_unsafe2)]
    plt.plot(full_unsafe2[:, 0], full_unsafe2[:, 1])
    full_unsafe3 = gen_full_data(prob.UNSAFE_3, superp.DATA_LEN_U / 2)
    full_unsafe3 = full_unsafe3[:len(full_unsafe3)]
    plt.plot(full_unsafe3[:, 0], full_unsafe3[:, 1])
    full_unsafe4 = gen_full_data(prob.UNSAFE_4, superp.DATA_LEN_U / 2)
    full_unsafe4 = full_unsafe4[: len(full_unsafe4)]
    plt.plot(full_unsafe4[:, 0], full_unsafe4[:, 1])
    # full_unsafe = gen_full_data(prob.UNSAFE, superp.DATA_LEN_U)
    # full_unsafe = full_unsafe[:len(full_unsafe)]
    # plt.plot(full_unsafe[:, 0], full_unsafe[:, 1])
    plt.show()
    # full_unsafe = data_filter(prob.INIT, full_unsafe)
    full_unsafe = torch.cat([full_unsafe1, full_unsafe2, full_unsafe3, full_unsafe4], dim=0)

    full_domain = gen_full_data(prob.DOMAIN, superp.DATA_LEN_D)

    def batch_data(full_data, data_length, data_chunks, filter):
        l = list(data_length)
        batch_list = [torch.reshape(full_data, l + [prob.DIM])]
        for i in range(prob.DIM):
            batch_list = [tensor_block for curr_tensor in batch_list for tensor_block in list(curr_tensor.chunk(int(data_chunks[i]), i))]
        
        batch_list = [torch.reshape(curr_tensor, [-1, prob.DIM]) for curr_tensor in batch_list]
        batch_list = [curr_tensor[filter(curr_tensor)] for curr_tensor in batch_list]

        return batch_list

    batch_init = batch_data(full_init, superp.DATA_LEN_I, superp.BLOCK_LEN_I, prob.cons_init)
    batch_unsafe = batch_data(full_unsafe, superp.DATA_LEN_U, superp.BLOCK_LEN_U, prob.cons_unsafe)
    batch_domain = batch_data(full_domain, superp.DATA_LEN_D, superp.BLOCK_LEN_D, prob.cons_domain)

    return batch_init, batch_unsafe, batch_domain
