# -*- coding: utf-8 -*-


"""
One-pulse single-qubit gate.
"""
from renom_q.circuit import Gate
from renom_q.circuit import QuantumCircuit
from renom_q.circuit import QuantumRegister
from renom_q.circuit.decorators import _1q_gate
from renom_q.converters.dagcircuit import DAGCircuit
from renom_q.qasm import pi
from . import header  # pylint: disable=unused-import
from .ubase import UBase


class U2Gate(Gate):
    """One-pulse single-qubit gate."""

    def __init__(self, phi, lam, qubit, circ=None):
        """Create new one-pulse single-qubit gate."""
        super().__init__("u2", [phi, lam], [qubit], circ)

    def _define_decompositions(self):
        decomposition = DAGCircuit()
        q = QuantumRegister(1, "q")
        decomposition.add_qreg(q)
        decomposition.add_basis_element("U", 1, 0, 3)
        rule = [
            UBase(pi / 2, self.params[0], self.params[1], q[0])
        ]
        for inst in rule:
            decomposition.apply_operation_back(inst)
        self._decompositions = [decomposition]

    def inverse(self):
        """Invert this gate.

        u2(phi,lamb)^dagger = u2(-lamb-pi,-phi+pi)
        """
        phi = self.params[0]
        self.params[0] = -self.params[1] - pi
        self.params[1] = -phi + pi
        self._decompositions = None
        return self

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.u2(self.params[0], self.params[1], self.qargs[0]))


@_1q_gate
def u2(self, phi, lam, q):
    """Apply u2 to q."""
    self._check_qubit(q)
    return self._attach(U2Gate(phi, lam, q, self))


QuantumCircuit.u2 = u2
