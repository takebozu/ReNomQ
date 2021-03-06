# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
controlled-H gate.
"""
from renom_q.circuit import Gate
from renom_q.circuit import QuantumCircuit
from renom_q.circuit import QuantumRegister
from renom_q.circuit.decorators import _control_target_gate
from renom_q.converters.dagcircuit import DAGCircuit
from . import header  # pylint: disable=unused-import
from .x import XGate
from .h import HGate
from .cx import CnotGate
from .t import TGate
from .s import SGate
from .s import SdgGate


class CHGate(Gate):
    """controlled-H gate."""

    def __init__(self, ctl, tgt, circ=None):
        """Create new CH gate."""
        super().__init__("ch", [], [ctl, tgt], circ)

    def _define_decompositions(self):
        """
        gate ch a,b {
        h b;
        sdg b;
        cx a,b;
        h b;
        t b;
        cx a,b;
        t b;
        h b;
        s b;
        x b;
        s a;}
        """
        decomposition = DAGCircuit()
        q = QuantumRegister(2, "q")
        decomposition.add_qreg(q)
        decomposition.add_basis_element("x", 1, 0, 0)
        decomposition.add_basis_element("h", 1, 0, 0)
        decomposition.add_basis_element("cx", 2, 0, 0)
        decomposition.add_basis_element("t", 1, 0, 0)
        decomposition.add_basis_element("s", 1, 0, 0)
        decomposition.add_basis_element("sdg", 1, 0, 0)
        rule = [
            HGate(q[1]),
            SdgGate(q[1]),
            CnotGate(q[0], q[1]),
            HGate(q[1]),
            TGate(q[1]),
            CnotGate(q[0], q[1]),
            TGate(q[1]),
            HGate(q[1]),
            SGate(q[1]),
            XGate(q[1]),
            SGate(q[0])
        ]
        for inst in rule:
            decomposition.apply_operation_back(inst)
        self._decompositions = [decomposition]

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.ch(self.qargs[0], self.qargs[1]))


@_control_target_gate
def ch(self, ctl, tgt):
    """Apply CH from ctl to tgt."""
    self._check_qubit(ctl)
    self._check_qubit(tgt)
    self._check_dups([ctl, tgt])
    return self._attach(CHGate(ctl, tgt, self))


QuantumCircuit.ch = ch
