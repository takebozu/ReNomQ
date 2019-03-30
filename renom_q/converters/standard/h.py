# -*- coding: utf-8 -*-


"""
Hadamard gate.
"""
from renom_q.circuit import Gate
from renom_q.circuit import QuantumCircuit
from renom_q.circuit import QuantumRegister
from renom_q.circuit.decorators import _1q_gate
from renom_q.qasm import pi
from renom_q.converters.dagcircuit import DAGCircuit
from . import header  # pylint: disable=unused-import
from .u2 import U2Gate


class HGate(Gate):
    """Hadamard gate."""

    def __init__(self, qubit, circ=None):
        """Create new Hadamard gate."""
        super().__init__("h", [], [qubit], circ)

    def _define_decompositions(self):
        """
        gate h a { u2(0,pi) a; }
        """
        decomposition = DAGCircuit()
        q = QuantumRegister(1, "q")
        decomposition.add_qreg(q)
        decomposition.add_basis_element("u2", 1, 0, 2)
        rule = [
            U2Gate(0, pi, q[0])
        ]
        for inst in rule:
            decomposition.apply_operation_back(inst)
        self._decompositions = [decomposition]

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.h(self.qargs[0]))


@_1q_gate
def h(self, q):
    """Apply H to q."""
    self._check_qubit(q)
    return self._attach(HGate(q, self))


QuantumCircuit.h = h
