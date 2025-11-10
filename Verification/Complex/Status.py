import sys, os

import torch

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))
#from Modules.utils import *
#from Modules.NNet import NeuralNetwork as NNet
#import Scripts.PARA as PARA


class NeuronStatus:
    '''Class to store the status of a neuron in a network
    Inputs:
    layer: int, layer index
    neuron: int, neuron index
    status: int, status of the neuron
    Features:
    get_id: returns the layer and neuron index
    get_status: returns the status of the neuron
    set_status: sets the status of the neuron
    set_status_from_value: sets the status of the neuron based on the value
    display: prints the layer, neuron and status of the neuron
    '''

    def __init__(self, layer: int, neuron: int, status: int):
        self.layer = layer
        self.neuron = neuron
        # status: -2: unknown, -1: negative, 0: zero, 1: positive
        self.status = status if status is not None else -2
        self.tol = 1e-12

    def get_id(self):
        return [self.layer, self.neuron]

    def get_status(self):
        return self.status

    def set_status(self, new_status: int) -> None:
        self.status = new_status

    def set_status_from_bound(self,
                              value_lower: torch.tensor, value_upper: torch.tensor) -> None:
        if value_lower > self.tol:
            self.status = 1
        elif value_upper < -self.tol:
            self.status = -1
        else:
            self.status = 0

    def set_status_from_value(self,
                              value:torch.tensor) -> None:
        if value > self.tol:
            self.status = 1
        elif value < -self.tol:
            self.status = -1
        else:
            self.status = 0

    def display(self):
        print("Layer: ", self.layer, " Neuron: ", self.neuron, " Status: ", self.status)


class NetworkStatus:
    ''' Class to store the status a given network
    Inputs:
    network: NNet, neural network
    Features:
    set_layer_status: sets the status of the layer
    set_layer_status_from_value: sets the status of the layer from the value
    get_neuron_inputs: returns the input to the neurons
    display_layer_status: prints the status of the layer
    get_netstatus_from_input: gets the network status from the input
    display_network_status_value: prints the network status
    '''

    def __init__(self, network):
        self.network = network.to('cpu')
        self.neuron_inputs = {}
        self.network_status = {}
        self.network_status_values = {}
        self.network_status_valuesMatrix = None

    def set_layer_status(self, layer_idx: int, layer_status: int):
        for neuron_idx, neuron_status in enumerate(layer_status):
            if self.network_status == {}:
                self.network_status = [NeuronStatus(layer_idx, status_item, -2) for status_item in layer_status]
            else:
                for status_idx, status_item in enumerate(layer_status):
                    self.network_status[status_idx].set_status(status_item)

    def set_layer_status_from_bound(self, layer_idx, lower_bound, upper_bound) -> list:
        layer_status = list([])
        for neuron_idx, neuron_input in enumerate(lower_bound):
            neuron_status = NeuronStatus(int(layer_idx / 2), int(neuron_idx), -2)
            neuron_status.set_status_from_bound(neuron_input, upper_bound[neuron_idx])
            # neuron_status.display()
            layer_status.append(neuron_status)
        return layer_status

    def set_layer_status_from_value(self, layer_idx, input_value) -> list:
        layer_status = list([])
        for neuron_idx, neuron_input in enumerate(input_value):
            neuron_status = NeuronStatus(int(layer_idx/2), int(neuron_idx), -2)
            neuron_status.set_status_from_value(neuron_input)
            # neuron_status.display()
            layer_status.append(neuron_status)
        return layer_status

    def get_neuron_inputs(self) -> dict:
        return self.neuron_inputs

    def display_layer_status(self, layer_status: list):
        print("Layer Status: ", [nstatus.status for nstatus in layer_status])

    def get_netstatus_from_input(self,
                                 input_value: torch.tensor) -> None:
        x = torch.tensor(input_value)
        for layer_idx, layer in enumerate(self.network):
            x = layer(x)
            # if layer is even, then it is a linear layer not activation layer
            if layer_idx % 2 == 0:  # starting from 0
                self.neuron_inputs[int(layer_idx / 2)] = x.tolist()
                layer_status = self.set_layer_status_from_value(layer_idx, x)
                self.network_status[int(layer_idx / 2)] = layer_status
                self.network_status_values[int(layer_idx / 2)] = [nstatus.status for nstatus in layer_status]
        # print('Propagation completed.')
        # self.display_network_status_value()

    def get_netstatus_from_input_bound(self,
                                 l_input: torch.tensor, r_input: torch.tensor) -> None:
        x1 = l_input
        x2 = r_input

        upper_bound = torch.max(x1, x2)
        lower_bound = torch.min(x1, x2)

        l = len(list(enumerate(self.network)))
        for layer_idx, layer in enumerate(self.network):
            if layer_idx % 2 == 0: 
                miu_pre = (upper_bound + lower_bound) / 2
                r_pre = (upper_bound - lower_bound) / 2
                miu = layer(miu_pre)
                r_k = r_pre @ torch.abs(layer.weight).T
                upper_bound = miu + r_k
                lower_bound = miu - r_k
                layer_status = self.set_layer_status_from_bound(layer_idx, lower_bound, upper_bound)
                self.network_status[int(layer_idx / 2)] = layer_status
                self.network_status_values[int(layer_idx / 2)] = [nstatus.status for nstatus in layer_status]
            else:
                upper_bound = layer(upper_bound)
                lower_bound = layer(lower_bound)


    def display_network_status_value(self):
        print("Network Status: ", self.network_status_values)


if __name__ == '__main__':
    architecture = [('relu', 2), ('relu', 32), ('linear', 1)]
    model = NNet(architecture)
    NStatus = NetworkStatus(model)

    # Generate random input using torch.rand for the model
    input_size = model.layers[0].in_features
    random_input = torch.rand(input_size)
    x = random_input
    # NStatus.forward_propagation(x)
    NStatus.get_netstatus_from_input(x)

