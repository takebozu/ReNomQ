# -*- coding: utf-8 -*-


"""
Rotation around the z-axis.
"""
from renom_q.circuit import Gate
from renom_q.circuit import QuantumCircuit
from renom_q.circuit import QuantumRegister
from renom_q.circuit.decorators import _1q_gate
from renom_q.converters.dagcircuit import DAGCircuit
from . import header  # pylint: disable=unused-import
from .u1 import U1Gate


class RZGate(Gate):
    """rotation around the z-axis."""

    def __init__(self, phi, qubit, circ=None):
        """Create new rz single qubit gate."""
        super().__init__("rz", [phi], [qubit], circ)

    def _define_decompositions(self):
        """
        gate rz(phi) a { u1(phi) a; }
        """
        decomposition = DAGCircuit()
        q = QuantumRegister(1, "q")
        decomposition.add_qreg(q)
        decomposition.add_basis_element("u1", 1, 0, 1)
        rule = [
            U1Gate(self.params[0], q[0])
        ]
        for inst in rule:
            decomposition.apply_operation_back(inst)
        self._decompositions = [decomposition]

    def inverse(self):
        """Invert this gate.

        rz(phi)^dagger = rz(-phi)
        """
        self.params[0] = -self.params[0]
        self._decompositions = None
        return self

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.rz(self.params[0], self.qargs[0]))


@_1q_gate
def rz(self, phi, q):
    """Apply Rz to q."""
    self._check_qubit(q)
    return self._attach(RZGate(phi, q, self))


QuantumCircuit.rz = rz
