import torch
import superp

############################################
# set default data type to double
############################################
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_dtype(torch.float32)
# torch.set_default_tensor_type(torch.FloatTensor)


############################################
# set the system dimension
############################################
DIM = 6


############################################
# set the super-rectangle range
############################################
# set the initial in super-rectangle
INIT = [[-0.2, 0.2], \
            [-0.2, 0.2], \
                [-0.2, 0.2], \
                    [-0.2, 0.2], \
                        [-0.2, 0.2], \
                            [-0.2, 0.2]
        ]


INIT_SHAPE = 1 # 1 for rectangle/cube; 2 for cycle/sphere



# the the unsafe in super-rectangle
UNSAFE = [[-1, 1], \
            [-1, 1], \
                [-1, 1],\
                    [-1, 1], \
                        [-1, 1], \
                            [-1, 1]
        ]

UNSAFE_1 = [[-1, -0.5], \
            [-1, 1], \
            [-1, 1]
            ]

UNSAFE_2 = [[0.5, 1], \
            [-1, 1], \
            [-1, 1], \
            ]

UNSAFE_3 = [[-1, 1], \
            [-1, -0.5], \
            [-1, 1]
            ]

UNSAFE_4 = [[-1, 1], \
            [0.5, 1], \
            [-1, 1], \
            ]

UNSAFE_5 = [[-1, 1], \
            [-1, 1], \
            [-1, -0.5], \
            ]

UNSAFE_6 = [[-1, 1], \
            [-1, 1], \
            [0.5, 1], \
            ]

UNSAFE_7 = [[-1, 1], \
            [-1, 1], \
            [-1.3, -1], \
            ]

UNSAFE_8 = [[-1, 1], \
            [-1, 1], \
            [1, 1.3], \
            ]




UNSAFE_SHAPE = 3 # 1 for rectangle/cube; 2 for cycle/sphere

SUB_UNSAFE = []
SUB_UNSAFE_SHAPE = []

# SUB_UNSAFE = [ # [[-1.2, -0.8], [0.3, 0.7], [-2, 2]], \
#                 [[-0.2, 0.2], [-1.2, -0.8], [-2, 2] ]
# ]
# SUB_UNSAFE_SHAPE = [3, 3, 3] # 3 for cylinder


# the the domain in super-rectangle
DOMAIN = [[-2, 2], \
            [-2, 2], \
                [-2, 2], \
[-2, 2], \
            [-2, 2], \
                [-2, 2]
        ]

DOMAIN_SHAPE = 1

############################################
# set the range constraints
############################################
def cons_init(x): # accept a two-dimensional tensor and return a tensor of bool with the same number of columns
    return x[:, 0] == x[:, 0] # equivalent to True

def cons_unsafe(x):
    # return x[:, 0] == x[:, 0]
    unsafe = (x[:, 0] - 0) * (x[:, 0] - 0) + (x[:, 1] + 0) * (x[:, 1] + 0) + (x[:, 2] + 0) * (x[:, 2] + 0) + (x[:, 3] + 0) * (x[:, 3] + 0) + (x[:, 4] + 0) * (x[:, 4] + 0) + (x[:, 5] + 0) * (x[:, 5] + 0)>= 0.25 + superp.TOL_DATA_GEN # a cylinder
#     # return unsafe

def cons_domain(x):
    return x[:, 0] == x[:, 0] # equivalent to True


############################################
# set the vector field
############################################
# this function accepts a tensor input and returns the vector field of the same size
def vector_field(x):
    def f(i, x):
        if i == 1:
            return -x[:, 0] * (1 + x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2 + x[:, 3] ** 2 + x[:, 4] ** 2 + x[:, 5] ** 2) # x[:, 1] stands for x2
        elif i == 2:
            return -x[:, 1] * (1 + x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2 + x[:, 3] ** 2 + x[:, 4] ** 2 + x[:, 5] ** 2)  # x[:, 0] stands for x1
        elif i == 3:
            # return -x[:, 0] - 2 * x[:, 1] - x[:, 2] + x[:, 0] ** 3
            return -x[:, 2] * (1 + x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2 + x[:, 3] ** 2 + x[:, 4] ** 2 + x[:, 5] ** 2)
        elif i == 4:
            return -x[:, 3] * (1 + x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2 + x[:, 3] ** 2 + x[:, 4] ** 2 + x[:, 5] ** 2)
        elif i == 5:
            return -x[:, 4] * (1 + x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2 + x[:, 3] ** 2 + x[:, 4] ** 2 + x[:, 5] ** 2)
        elif i == 6:
            return -x[:, 5] * (1 + x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2 + x[:, 3] ** 2 + x[:, 4] ** 2 + x[:, 5] ** 2)
        else:
            print("Vector function error!")
            exit()

    vf = torch.stack([f(i + 1, x) for i in range(DIM)], dim=1)
    return vf
