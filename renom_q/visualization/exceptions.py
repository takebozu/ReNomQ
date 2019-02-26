# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Exceptions for errors raised by ReNomQ."""


class ReNomQError(Exception):
    """Base class for errors raised by ReNomQ."""

    def __init__(self, *message):
        """Set the error message."""
        super().__init__(' '.join(message))
        self.message = ' '.join(message)

    def __str__(self):
        """Return the message."""
        return repr(self.message)


class ReNomQIndexError(ReNomQError, IndexError):
    """Raised when a sequence subscript is out of range."""
    pass

class VisualizationError(ReNomQError):
    """For visualization specific errors."""
    pass

class DAGCircuitError(ReNomQError):
    """Base class for errors raised by the DAGCircuit object."""

    def __init__(self, *msg):
        """Set the error message."""
        super().__init__(*msg)
        self.msg = ' '.join(msg)

    def __str__(self):
        """Return the message."""
        return repr(self.msg)
class QasmError(ReNomQError):
    """Base class for errors raised while parsing OPENQASM."""

    def __init__(self, *msg):
        """Set the error message."""
        super().__init__(*msg)
        self.msg = ' '.join(msg)

    def __str__(self):
        """Return the message."""
        return repr(self.msg)
