# -*- coding: utf-8 -*-


"""
Node for an OPENQASM integer.
"""
from sympy import N

from ._node import Node


class Int(Node):
    """Node for an OPENQASM integer.

    This node has no children. The data is in the value field.
    """

    def __init__(self, id):
        """Create the integer node."""
        # pylint: disable=redefined-builtin
        Node.__init__(self, "int", None, None)
        self.value = id

    def to_string(self, indent):
        """Print with indent."""
        ind = indent * ' '
        print(ind, 'int', self.value)

    def qasm(self, prec=15):
        """Return the corresponding OPENQASM string."""
        # pylint: disable=unused-argument
        return "%d" % self.value

    def latex(self, prec=15, nested_scope=None):
        """Return the corresponding math mode latex string."""
        # pylint: disable=unused-argument
        return "%d" % self.value

    def sym(self, nested_scope=None):
        """Return the correspond symbolic number."""
        # pylint: disable=unused-argument
        return N(self.value)

    def real(self, nested_scope=None):
        """Return the correspond floating point number."""
        # pylint: disable=unused-argument
        return float(self.value)
