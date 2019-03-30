# -*- coding: utf-8 -*-


"""
Element of SU(2).
"""
from renom_q.circuit import Gate
from renom_q.circuit import QuantumCircuit
from renom_q.circuit.decorators import _1q_gate
from . import header  # pylint: disable=unused-import


class UBase(Gate):  # pylint: disable=abstract-method
    """Element of SU(2)."""

    def __init__(self, theta, phi, lam, qubit, circ=None):
        super().__init__("U", [theta, phi, lam], [qubit], circ)

    def inverse(self):
        """Invert this gate.

        U(theta,phi,lambda)^dagger = U(-theta,-lambda,-phi)
        """
        self.params[0] = -self.params[0]
        phi = self.params[1]
        self.params[1] = -self.params[2]
        self.params[2] = -phi
        return self

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.u_base(self.params[0], self.params[1], self.params[2],
                                    self.qargs[0]))


@_1q_gate
def u_base(self, theta, phi, lam, q):
    """Apply U to q."""
    self._check_qubit(q)
    return self._attach(UBase(theta, phi, lam, q, self))


QuantumCircuit.u_base = u_base
