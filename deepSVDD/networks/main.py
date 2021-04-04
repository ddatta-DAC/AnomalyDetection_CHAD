import os

try:
    from .AE import AE
except:
    from AE import AE

try:
    from .AE import FC_dec
    from .AE import FC_enc
except:
    from AE import FC_dec
    from AE import FC_enc

def build_network(fc_layer_dims):
    """Builds the neural network."""
    # implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU','FC')
    # assert net_name in implemented_networks
    net = FC_enc(fc_layer_dims)
    return net


def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU')
    assert net_name in implemented_networks

    ae_net = None

    return ae_net
