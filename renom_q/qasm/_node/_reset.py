# -*- coding: utf-8 -*-


"""
Node for an OPENQASM reset statement.
"""
from ._node import Node


class Reset(Node):
    """Node for an OPENQASM reset statement.

    children[0] is a primary node (id or indexedid)
    """

    def __init__(self, children):
        """Create the reset node."""
        Node.__init__(self, 'reset', children, None)

    def qasm(self, prec=15):
        """Return the corresponding OPENQASM string."""
        return "reset " + self.children[0].qasm(prec) + ";"
