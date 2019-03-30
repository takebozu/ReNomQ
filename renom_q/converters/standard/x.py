# -*- coding: utf-8 -*-


"""
Pauli X (bit-flip) gate.
"""
from renom_q.circuit import Gate
from renom_q.circuit import QuantumCircuit
from renom_q.circuit import QuantumRegister
from renom_q.circuit.decorators import _1q_gate
from renom_q.converters.dagcircuit import DAGCircuit
from renom_q.qasm import pi
from . import header  # pylint: disable=unused-import
from .u3 import U3Gate


class XGate(Gate):
    """Pauli X (bit-flip) gate."""

    def __init__(self, qubit, circ=None):
        """Create new X gate."""
        super().__init__("x", [], [qubit], circ)

    def _define_decompositions(self):
        """
        gate x a {
        u3(pi,0,pi) a;
        }
        """
        decomposition = DAGCircuit()
        q = QuantumRegister(1, "q")
        decomposition.add_qreg(q)
        decomposition.add_basis_element("u3", 1, 0, 3)
        rule = [
            U3Gate(pi, 0, pi, q[0])
        ]
        for inst in rule:
            decomposition.apply_operation_back(inst)
        self._decompositions = [decomposition]

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.x(self.qargs[0]))


@_1q_gate
def x(self, q):
    """Apply X to q."""
    self._check_qubit(q)
    return self._attach(XGate(q, self))


QuantumCircuit.x = x
