# -*- coding: utf-8 -*-


"""
controlled-rz gate.
"""
from renom_q.circuit import Gate
from renom_q.circuit import QuantumCircuit
from renom_q.circuit import QuantumRegister
from renom_q.circuit.decorators import _control_target_gate
from renom_q.converters.dagcircuit import DAGCircuit
from . import header  # pylint: disable=unused-import
from .u1 import U1Gate
from .cx import CnotGate


class CrzGate(Gate):
    """controlled-rz gate."""

    def __init__(self, theta, ctl, tgt, circ=None):
        """Create new crz gate."""
        super().__init__("crz", [theta], [ctl, tgt], circ)

    def _define_decompositions(self):
        """
        gate crz(lambda) a,b
        { u1(lambda/2) b; cx a,b;
          u1(-lambda/2) b; cx a,b;
        }
        """
        decomposition = DAGCircuit()
        q = QuantumRegister(2, "q")
        decomposition.add_qreg(q)
        decomposition.add_basis_element("u1", 1, 0, 1)
        decomposition.add_basis_element("cx", 2, 0, 0)
        rule = [
            U1Gate(self.params[0] / 2, q[1]),
            CnotGate(q[0], q[1]),
            U1Gate(-self.params[0] / 2, q[1]),
            CnotGate(q[0], q[1])
        ]
        for inst in rule:
            decomposition.apply_operation_back(inst)
        self._decompositions = [decomposition]

    def inverse(self):
        """Invert this gate."""
        self.params[0] = -self.params[0]
        self._decompositions = None
        return self

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.crz(self.params[0], self.qargs[0], self.qargs[1]))


@_control_target_gate
def crz(self, theta, ctl, tgt):
    """Apply crz from ctl to tgt with angle theta."""
    self._check_qubit(ctl)
    self._check_qubit(tgt)
    self._check_dups([ctl, tgt])
    return self._attach(CrzGate(theta, ctl, tgt, self))


QuantumCircuit.crz = crz
