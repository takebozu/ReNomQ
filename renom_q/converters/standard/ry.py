# -*- coding: utf-8 -*-


"""
Rotation around the y-axis.
"""
from renom_q.circuit import Gate
from renom_q.circuit import QuantumCircuit
from renom_q.circuit import QuantumRegister
from renom_q.circuit.decorators import _1q_gate
from renom_q.converters.dagcircuit import DAGCircuit
from . import header  # pylint: disable=unused-import
from .u3 import U3Gate


class RYGate(Gate):
    """rotation around the y-axis."""

    def __init__(self, theta, qubit, circ=None):
        """Create new ry single qubit gate."""
        super().__init__("ry", [theta], [qubit], circ)

    def _define_decompositions(self):
        """
        gate ry(theta) a { u3(theta, 0, 0) a; }
        """
        decomposition = DAGCircuit()
        q = QuantumRegister(1, "q")
        decomposition.add_qreg(q)
        decomposition.add_basis_element("u3", 1, 0, 3)
        rule = [
            U3Gate(self.params[0], 0, 0, q[0])
        ]
        for inst in rule:
            decomposition.apply_operation_back(inst)
        self._decompositions = [decomposition]

    def inverse(self):
        """Invert this gate.

        ry(theta)^dagger = ry(-theta)
        """
        self.params[0] = -self.params[0]
        self._decompositions = None
        return self

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.ry(self.params[0], self.qargs[0]))


@_1q_gate
def ry(self, theta, q):
    """Apply Ry to q."""
    self._check_qubit(q)
    return self._attach(RYGate(theta, q, self))


QuantumCircuit.ry = ry
