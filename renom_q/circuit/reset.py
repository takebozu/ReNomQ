# -*- coding: utf-8 -*-


"""
Qubit reset to computational zero.
"""
from .instruction import Instruction
from .instructionset import InstructionSet
from .quantumcircuit import QuantumCircuit
from .quantumregister import QuantumRegister


class Reset(Instruction):
    """Qubit reset."""

    def __init__(self, qubit, circ=None):
        """Create new reset instruction."""
        super().__init__("reset", [], [qubit], [], circ)

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.reset(self.qargs[0]))


def reset(self, quantum_register):
    """Reset q."""
    if isinstance(quantum_register, QuantumRegister):
        instructions = InstructionSet()
        for sizes in range(quantum_register.size):
            instructions.add(self.reset((quantum_register, sizes)))
        return instructions
    self._check_qubit(quantum_register)
    return self._attach(Reset(quantum_register, self))


QuantumCircuit.reset = reset
