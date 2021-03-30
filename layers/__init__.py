from layers.low_rank import LowRank
from layers.basic_att import BasicAtt
from layers.sc_att import SCAtt
from layers.capsule_low_rank import CapsuleLowRank
from layers.capsules import Capsule

__factory = {   
    'LowRank': LowRank,
    'CapsuleLowRank': CapsuleLowRank,
    'BasicAtt': BasicAtt,
    'SCAtt': SCAtt,
    'Capsule': Capsule
}

def names():
    return sorted(__factory.keys())

def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown layer:", name)
    return __factory[name](*args, **kwargs)