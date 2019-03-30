# -*- coding: utf-8 -*-


"""
Node for an OPENQASM CNOT statement.
"""
from ._node import Node


class Cnot(Node):
    """Node for an OPENQASM CNOT statement.

    children[0], children[1] are id nodes if CX is inside a gate body,
    otherwise they are primary nodes.
    """

    def __init__(self, children):
        """Create the cnot node."""
        Node.__init__(self, 'cnot', children, None)

    def qasm(self, prec=15):
        """Return the corresponding OPENQASM string."""
        return "CX " + self.children[0].qasm(prec) + "," + \
               self.children[1].qasm(prec) + ";"
