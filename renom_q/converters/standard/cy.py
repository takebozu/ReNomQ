# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
controlled-Y gate.
"""
from circuit import Gate
from circuit import QuantumCircuit
from circuit import QuantumRegister
from circuit.decorators import _control_target_gate
from converters.dagcircuit import DAGCircuit
from . import header  # pylint: disable=unused-import
from .s import SGate
from .s import SdgGate
from .cx import CnotGate


class CyGate(Gate):
    """controlled-Y gate."""

    def __init__(self, ctl, tgt, circ=None):
        """Create new CY gate."""
        super().__init__("cy", [], [ctl, tgt], circ)

    def _define_decompositions(self):
        """
        gate cy a,b { sdg b; cx a,b; s b; }
        """
        decomposition = DAGCircuit()
        q = QuantumRegister(2, "q")
        decomposition.add_qreg(q)
        decomposition.add_basis_element("s", 1, 0, 0)
        decomposition.add_basis_element("sdg", 1, 0, 0)
        decomposition.add_basis_element("cx", 2, 0, 0)
        rule = [
            SdgGate(q[1]),
            CnotGate(q[0], q[1]),
            SGate(q[1])
        ]
        for inst in rule:
            decomposition.apply_operation_back(inst)
        self._decompositions = [decomposition]

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.cy(self.qargs[0], self.qargs[1]))


@_control_target_gate
def cy(self, ctl, tgt):
    """Apply CY to circuit."""
    self._check_qubit(ctl)
    self._check_qubit(tgt)
    self._check_dups([ctl, tgt])
    return self._attach(CyGate(ctl, tgt, self))


QuantumCircuit.cy = cy
