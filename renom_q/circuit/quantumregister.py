# -*- coding: utf-8 -*-


"""
Quantum register reference object.
"""
import itertools

from .register import Register


class QuantumRegister(Register):
    """Implement a quantum register."""
    # Counter for the number of instances in this class.
    instances_counter = itertools.count()
    # Prefix to use for auto naming.
    prefix = 'q'

    def qasm(self):
        """Return OPENQASM string for this register."""
        return "qreg %s[%d];" % (self.name, self.size)
