# -*- coding: utf-8 -*-

import numpy as np
import collections
import copy
import matplotlib.pyplot as plt
from renom_q.visualization._circuit_visualization import matplotlib_circuit_drawer as drawer_mpl
from renom_q.visualization._circuit_visualization import _text_circuit_drawer as drawer_text
from renom_q.visualization._circuit_visualization import _latex_circuit_drawer as drawer_latext
from renom_q.visualization._circuit_visualization import circuit_drawer
from renom_q.circuit.quantumcircuit import QuantumCircuit as q_ctl


class QuantumRegister:
    """ Definite a quantum register. """

    def __init__(self, num, name=None):
        """
        Args:
            num (int):
                The number of quantum bits.
            name (str):
                A name of quantum bits. Defaults to 'q'.

        Example:
            >>> import renom_q
            >>> q = renom_q.QuantumRegister(1)
        """
        self.name = 'q' if name is None else name
        self.dict = {self.name: 0}
        self.num = num
        self.numlist = [num]
        self.q_states = 2**num
        self.qubit = [complex(0.) for i in range(self.q_states)]
        self.qubit[0] = 1.
        self.q_number = [i for i in range(self.q_states)]
        self.qasmcode = 'qreg ' + str(self.name) + '[' + str(self.num) + '];'

    def __getitem__(self, num):
        """
        Args:
            num (int):
                The quantum bit number. If definited 3 quantum bits,
                most significant bit number is 0 and least significant bit
                number is 2.

        Returns:
            (tuple):
                A tuple of a name of quantum bits and the quantum bit number.

        Example:
            >>> import renom_q
            >>> q = renom_q.QuantumRegister(1, 'qbit')
            >>> q[0]
            ('qbit', 0)
        """
        return self.name, num

    def __str__(self):
        """
        Returns:
            (str): A name of quantum bits.

        Example:
            >>> import renom_q
            >>> q = renom_q.QuantumRegister(1, 'qbit')
            >>> print(q)
            qbit
        """
        return self.name


class ClassicalRegister:
    """ Definite a classical register. """

    def __init__(self, num, name=None):
        """
        Args:
            num (int):
                The number of classical bits.
            name (str):
                A name of classical bits. Defaults to 'c'.

        Example:
            >>> import renom_q
            >>> c = renom_q.ClassicalRegister(1)
        """
        self.name = 'c' if name is None else name
        self.dict = {self.name: 0}
        self.num = num
        self.numlist = [num]
        self.cbit = [0 for i in range(self.num)]
        self.qasmcode = 'creg ' + str(self.name) + '[' + str(self.num) + '];'

    def __getitem__(self, num):
        """
        Args:
            num (int):
                The classical bit number. If definited 3 classical bits,
                most significant bit numberis 0 and least significant bit
                number is 2.

        Returns:
            (tuple):
                A tuple of a name of classical bits and the classical bit number.

        Example:
            >>> import renom_q
            >>> c = renom_q.ClassicalRegister(1, 'cbit')
            >>> c[0]
            ('cbit', 0)
        """
        return self.name, num

    def __str__(self):
        """
        Returns:
            (str):
            A name of classical bits.

        Example:
            >>> import renom_q
            >>> c = renom_q.ClassicalRegister(1, 'cbit')
            >>> print(c)
            cbit
        """
        return self.name


def load_qasm_string(qasm_string, name=None):
    qc = q_ctl.from_qasm_str(qasm_string)
    if name:
        qc.name = name
    return qc


def plot_histogram(Counts):
    """ Plot the execution result with histogram.

    Args:
        Counts (dict):
            A dict of the execution result of quantum circuit mesurement.

    Returns:
        matplotlib.figure:
            A matplotlib figure object for the execution result of quantum
            circuit mesurement.

    Example:
        >>> import renom_q
        >>> q = renom_q.QuantumRegister(1)
        >>> c = renom_q.ClassicalRegister(1)
        >>> qc = renom_q.QuantumCircuit(q, c)
        >>> qc.measure()
        >>> r = renom_q.execute(qc)
        >>> renom_q.plot_histogram(r)
    """
    width = 0.5
    shots = np.sum([i for i in Counts.values()])
    x = [i[0] for i in Counts.items()]
    y = [i[1] / shots for i in Counts.items()]
    for i in range(len(Counts)):
        plt.text(x[i], y[i], str(y[i]), ha='center')
    plt.bar(x, y, width=width)
    plt.tick_params(top=1, right=1, direction='in')
    plt.xticks(rotation=60)
    plt.ylabel('Probabilities')
    plt.show()


def execute(QC, shots=1024):
    """ Execute the quantum circuit mesurement.

    Args:
        QC (renom_q.QuantumCircuit):
            A class of QuantumCircuit.
        shots (int):
            The number of excutions of quantum circuit mesurement. Defaults to
            1024.

    Returns:
        (dict):
            A execution result of quantum circuit mesurement. The key is the
            measured classical bits. The value is the number of classical bits
            measured.

    Example:
        >>> import renom_q
        >>> q = renom_q.QuantumRegister(1)
        >>> c = renom_q.ClassicalRegister(1)
        >>> qc = renom_q.QuantumCircuit(q, c)
        >>> qc.measure()
        >>> renom_q.execute(qc)
        {'0': 1024}
    """
    Code = ''
    Cn = []
    for i in QC.Codelist:
        Code += i
    else:
        Code += 'Cn.append(\'\'.join(QC.Cr.cbit));'
    # print(Code)
    for i in range(shots):
        exec(compile(Code, '<string>', 'exec'))

    return dict(sorted(collections.Counter(Cn).items(), key=lambda x: x[0]))


def print_matrix(QC, tensorgate=False):
    """ Print all matrix calculation of unitary conversion.

    Args:
        QC (renom_q.QuantumCircuit):
            A class of QuantumCircuit.
        tensorgate (bool):
            When set to True, added matrix calculation of quantum gate tensor
            product. Defaults to False.

    Returns:
        matrix(str):
            Strings of the final result of qubit statevector and all matrix
            calculation of unitary conversion.

    Example:
        >>> import renom_q
        >>> q = renom_q.QuantumRegister(1)
        >>> c = renom_q.ClassicalRegister(1)
        >>> qc = renom_q.QuantumCircuit(q, c)
        >>> qc.x(q[0])
        >>> renom_q.print_matrix(qc)
        ---------------- result qubit ----------------
        [0.+0.j 1.+0.j]
        ---------------- x(q[0]) ----------------
        [[0. 1.]
        [1. 0.]]・
                                                            \\
        [[1.+0.j]
        [0.+0.j]] =
                                                            \\
        [[0.+0.j]
        [1.+0.j]]
    """
    matrix = '---------------- result qubit ----------------\n' + str(QC.Qr.qubit)
    if tensorgate is True:
        matrixlist = np.array([QC.matrixlist, QC.tensorlist])
        for i in range(len(QC.matrixlist)):
            matrix += matrixlist[0, i]
            matrix += matrixlist[1, i]
    else:
        for i in QC.matrixlist:
            matrix += i
    print(matrix)


def draw_circuit(QC, style=None):
    """ Draw the quantum circuit diagram.

    Args:
        QC (renom_q.QuantumCircuit):
            A class of QuantumCircuit.
        style (dict or str):
            dictionary of style or file name of style file. Defaults to None.

            The style dict kwarg contains numerous options that define the style of the
            output circuit visualization. While the style dict is used by the `mpl`,
            `latex`, and `latex_source` outputs some options in that are only used
            by the `mpl` output. These options are defined below, if it is only used by
            the `mpl` output it is marked as such:

            textcolor (str):
                The color code to use for text. Defaults to `'#000000'` (`mpl` only)
            subtextcolor (str):
                The color code to use for subtext. Defaults to `'#000000'` (`mpl` only)
            linecolor (str):
                The color code to use for lines. Defaults to `'#000000'` (`mpl` only)
            creglinecolor (str):
                The color code to use for classical register lines `'#778899'`(`mpl` only)
            gatetextcolor (str):
                The color code to use for gate text `'#000000'` (`mpl` only)
            gatefacecolor (str):
                The color code to use for gates. Defaults to `'#ffffff'` (`mpl` only)
            barrierfacecolor (str):
                The color code to use for barriers. Defaults to `'#bdbdbd'` (`mpl` only)
            backgroundcolor (str):
                The color code to use for the background. Defaults to `'#ffffff'` (`mpl` only)
            fontsize (int):
                The font size to use for text. Defaults to 13 (`mpl` only)
            subfontsize (int):
                The font size to use for subtext. Defaults to 8 (`mpl` only)
            displaytext (dict):
                A dictionary of the text to use for each element type in the
                output visualization. The default values are:
                {
                    'id': 'id',
                    'u0': 'U_0',
                    'u1': 'U_1',
                    'u2': 'U_2',
                    'u3': 'U_3',
                    'x': 'X',
                    'y': 'Y',
                    'z': 'Z',
                    'h': 'H',
                    's': 'S',
                    'sdg': 'S^\\dagger',
                    't': 'T',
                    'tdg': 'T^\\dagger',
                    'rx': 'R_x',
                    'ry': 'R_y',
                    'rz': 'R_z',
                    'reset': '\\left|0\\right\\rangle'
                }
                You must specify all the necessary values if using this. There is
                no provision for passing an incomplete dict in. (`mpl` only)
            displaycolor (dict):
                The color codes to use for each circuit element. By default all
                values default to the value of `gatefacecolor` and the keys are
                the same as `displaytext`. Also, just like `displaytext` there
                is no provision for an incomplete dict passed in. (`mpl` only)
            latexdrawerstyle (bool):
                When set to True enable latex mode which will draw gates like
                the `latex` output modes. (`mpl` only)
            usepiformat (bool):
                When set to True use radians for output (`mpl` only)
            fold (int):
                The number of circuit elements to fold the circuit at. Defaults
                to 20 (`mpl` only)
            cregbundle (bool):
                If set True bundle classical registers (`mpl` only)
            showindex (bool):
                If set True draw an index. (`mpl` only)
            compress (bool):
                If set True draw a compressed circuit (`mpl` only)
            figwidth (int):
                The maximum width (in inches) for the output figure.
                (`mpl` only)
            dpi (int):
                The DPI to use for the output image. Defaults to 150 (`mpl` only)
            margin (list):
                `mpl` only
            creglinestyle (str):
                The style of line to use for classical registers. Choices are
                `'solid'`, `'doublet'`, or any valid matplotlib `linestyle` kwarg
                value. Defaults to `doublet`(`mpl` only)

    Returns:
        matplotlib.figure:
            A matplotlib figure object for the circuit diagram.

    Example:
        >>> import renom_q
        >>> q = renom_q.QuantumRegister(1)
        >>> c = renom_q.ClassicalRegister(1)
        >>> qc = renom_q.QuantumCircuit(q, c)
        >>> qc.measure()
        >>> renom_q.draw_circuit(qc)
    """
    qasm_string = 'OPENQASM 2.0;include "qelib1.inc";'
    for i in QC.qasmlist:
        qasm_string += i
    qasm = load_qasm_string(qasm_string)
    default_style = {'plotbarrier': False, 'creglinestyle': "solid",
                     'latexdrawerstyle': False, 'compress': True, 'usepiformat': False}
    input_style = default_style if style is None else style

    figure = drawer_mpl(qasm, style=input_style)
    return figure


def statevector(QC):
    """ Get the qubit statevector.

    Args:
        QC (renom_q.QuantumCircuit):
            A class of QuantumCircuit.

    Returns:
        (array):
            A array of the qubit statevector.

    Example:
        >>> import renom_q
        >>> q = renom_q.QuantumRegister(1)
        >>> c = renom_q.ClassicalRegister(1)
        >>> qc = renom_q.QuantumCircuit(q, c)
        >>> qc.x(q[0])
        >>> renom_q.statevector(qc)
        array([0.+0.j, 1.+0.j])
    """
    return QC.Qr.qubit


class QuantumCircuit(object):
    """ Definite a quantum circuit. """

    def __init__(self, *args, circuit_number=0):
        """
        Args:
            *args (renom_q.QuantumRegister and renom_q.ClassicalRegister):
                Quantum registers and classical registers that a quantum
                circuit consists of. In both registers, defining multiple
                registers is possible, but at least one register needed.
            circuit_number (int):
                The number used when conflating multiple quantum circuits. There
                is no need for the user to input.

        Example 1:
            >>> import renom_q
            >>> q = renom_q.QuantumRegister(1)
            >>> c = renom_q.ClassicalRegister(1)
            >>> qc = renom_q.QuantumCircuit(q, c)

        Example 2:
            >>> import renom_q
            >>> qa = renom_q.QuantumRegister(1, 'qa')
            >>> qb = renom_q.QuantumRegister(1, 'qb')
            >>> ca = renom_q.ClassicalRegister(1, 'ca')
            >>> cb = renom_q.ClassicalRegister(1, 'cb')
            >>> qc = renom_q.QuantumCircuit(qa, ca, qb, cb)

        Example 3:
            >>> import renom_q
            >>> qa = renom_q.QuantumRegister(1, 'qa')
            >>> qb = renom_q.QuantumRegister(1, 'qb')
            >>> ca = renom_q.ClassicalRegister(1, 'ca')
            >>> cb = renom_q.ClassicalRegister(1, 'cb')
            >>> qca = renom_q.QuantumCircuit(qa, ca)
            >>> qcb = renom_q.QuantumCircuit(qb, cb)
            >>> qc = qca + qcb
        """
        qr = []
        cr = []
        for i in range(len(args)):
            if type(args[i]) is QuantumRegister:
                qr.append(args[i])
            elif type(args[i]) is ClassicalRegister:
                cr.append(args[i])

        if len(qr) == 1:
            self.Qr = copy.deepcopy(qr[0])
        else:
            try:
                q_num = qr[0].num
                q_numlist = qr[0].numlist
                q_dict = qr[0].dict
                q_qasmcode = qr[0].qasmcode
                for i in range(1, len(qr)):
                    q_num += qr[i].num
                    q_numlist.extend(qr[i].numlist)
                    assert qr[i].name not in q_dict, qr[i].name + \
                        " is already used. Please use a different name."
                    q_dict.update([(qr[i].name, i)])
                    q_qasmcode += qr[i].qasmcode
                Q = QuantumRegister(q_num)
                Q.numlist = q_numlist
                Q.dict = q_dict
                Q.qasmcode = q_qasmcode
                self.Qr = copy.deepcopy(Q)
            except AssertionError as e:
                raise

        if len(cr) == 1:
            self.Cr = copy.deepcopy(cr[0])
        else:
            try:
                c_num = cr[0].num
                c_numlist = cr[0].numlist
                c_dict = cr[0].dict
                c_qasmcode = cr[0].qasmcode
                for i in range(1, len(cr)):
                    c_num += cr[i].num
                    c_numlist.extend(cr[i].numlist)
                    assert cr[i].name not in c_dict, cr[i].name + \
                        " is already used. Please use a different name."
                    c_dict.update([(cr[i].name, i)])
                    c_qasmcode += cr[i].qasmcode
                C = ClassicalRegister(c_num)
                C.numlist = c_numlist
                C.dict = c_dict
                C.qasmcode = c_qasmcode
                self.Cr = copy.deepcopy(C)
            except AssertionError as e:
                raise

        self.qubit = self.Qr.qubit.copy()
        self.circuit_number = circuit_number
        self.Codelist = []
        self.matrixlist = []
        self.tensorlist = []
        self.qasmlist = [self.Qr.qasmcode, self.Cr.qasmcode]
        self.idgate = np.array([[1., 0.], [0., 1.]])
        self.statevector = []
        self.measure_bool = True

    def __str__(self):
        """ Draw the quantum circuit diagram by texts.

        Returns:
            TextDrawing:
                An instances that, when printed, draws the circuit in ascii art.

        Example:
            >>> import renom_q
            >>> q = renom_q.QuantumRegister(2)
            >>> c = renom_q.ClassicalRegister(2)
            >>> qc = renom_q.QuantumCircuit(q, c)
            >>> qc.h(q[0])
            >>> qc.x(q[1])
            >>> qc.measure()
            >>> print(qc)
                            ┌───┐┌─┐
            q_0: |0>────────┤ H ├┤M├
                    ┌───┐┌─┐└───┘└╥┘
            q_1: |0>┤ X ├┤M├──────╫─
                    └───┘└╥┘      ║
             c_0: 0 ══════╬═══════╩═
                          ║
             c_1: 0 ══════╩═════════

        """
        return str(self._draw(output='text'))

    def _draw(self, scale=0.7,
              filename=None,
              style=None,
              output=None,
              interactive=False,
              line_length=None,
              plot_barriers=True,
              reverse_bits=False):
        qasm_string = 'OPENQASM 2.0;include "qelib1.inc";'
        for i in self.qasmlist:
            qasm_string += i
        qasm = load_qasm_string(qasm_string)
        return circuit_drawer(qasm, scale=0.7,
                              filename=filename,
                              style=style,
                              output=output,
                              interactive=interactive,
                              line_length=line_length,
                              plot_barriers=plot_barriers,
                              reverse_bits=reverse_bits)

    def __add__(self, other):
        try:
            q_num = self.Qr.num + other.Qr.num
            q_numlist = self.Qr.numlist
            q_numlist.extend(other.Qr.numlist)
            q_dict = self.Qr.dict
            assert other.Qr.name not in q_dict, other.Qr.name + \
                " is already used. Please use a different name."
            q_dict.update([(other.Qr.name, self.circuit_number + 1)])
            Q = QuantumRegister(q_num)
            Q.numlist = q_numlist
            Q.dict = q_dict
            Q.qasmcode = self.Qr.qasmcode + other.Qr.qasmcode

            c_dict = self.Cr.dict
            C = self.Cr
            if other.Cr.name not in c_dict:
                c_dict.update([(other.Cr.name, self.circuit_number + 1)])
                c_num = self.Cr.num + other.Cr.num
                c_numlist = self.Cr.numlist
                c_numlist.extend(other.Cr.numlist)
                C = ClassicalRegister(c_num)
                C.numlist = c_numlist
                C.dict = c_dict
                C.qasmcode = self.Cr.qasmcode + other.Cr.qasmcode

            return QuantumCircuit(Q, C, circuit_number=self.circuit_number + 1)
        except AssertionError as e:
            raise

    def barrier(self, *args):
        """ Add a barrier block in circuit diagram.

        args:
            *args (renom_q.QuantumRegister, tuple or None):
                If arg type is a tuple (ex: q[0]), a quantum register No.0 is
                added a barrier block. If arg type is a renom_q.QuantumRegister,
                all quantum registers in renom_q.QuantumRegister are added a
                barrier block. If arg type is None, all of multiple quantum
                registers are added a barrier block.

        Example:
            >>> import renom_q
            >>> q = renom_q.QuantumRegister(2)
            >>> c = renom_q.ClassicalRegister(2)
            >>> qc = renom_q.QuantumCircuit(q, c)
            >>> qc.h(q[0])
            >>> qc.barrier()
            >>> qc.x(q[1])
            >>> qc.measure()
            >>> print(qc)
                    ┌───┐ ░         ┌─┐
            q_0: |0>┤ H ├─░─────────┤M├
                    └───┘ ░ ┌───┐┌─┐└╥┘
            q_1: |0>──────░─┤ X ├┤M├─╫─
                          ░ └───┘└╥┘ ║
             c_0: 0 ══════════════╬══╩═
                                  ║
             c_1: 0 ══════════════╩════

        """
        STR = 'barrier '
        if args == ():
            for i in range(self.circuit_number + 1):
                for j in range(self.Qr.numlist[i]):
                    name = [k for k, v in self.Qr.dict.items() if v == i][0]
                    STR += str(name) + '[' + str(j) + '], '
        elif str(args[0]) in self.Qr.dict:
            for i in args:
                name = i.name
                idx = self.Qr.dict[name]
                for j in range(self.Qr.numlist[idx]):
                    STR += str(name) + '[' + str(j) + '], '
        elif type(args[0]) is tuple:
            for i in args:
                name, num = i
                STR += str(name) + '[' + str(num) + '], '

        STR = STR.rstrip(', ')
        STR += ';'
        # print(STR)
        self.qasmlist.append(STR)

        return self

    def measure_exec(self, statevector):
        qubit = statevector
        p = [i**2 for i in np.abs(qubit)]
        bit_list = list(format(np.random.choice(self.Qr.q_number, p=p),
                               '0' + str(self.Qr.num) + 'b'))
        return bit_list

    def measure(self, *args):
        """ Measure the quantum state of the qubits.

        args:
            *args (renom_q.QuantumRegister and renom_q.ClassicalRegister, tuple and tuple or None):
                If arg type is tuple and tuple (ex: q[0], c[0]), measured a quantum
                register No.0 into a classical register No.0. If arg type is a
                renom_q.QuantumRegister and a renom_q.ClassicalRegister, measured
                all quantum registers in renom_q.QuantumRegister into all classical
                registers in renom_q.ClassicalRegister. If arg is None, measured
                all of multiple quantum registers into all classical registers
                in renom_q.ClassicalRegister. Only when definited single classical
                register, coding like example1 is possible.

        Example 1:
            >>> import renom_q
            >>> q = renom_q.QuantumRegister(2)
            >>> c = renom_q.ClassicalRegister(2)
            >>> qc = renom_q.QuantumCircuit(q, c)
            >>> qc.h(q[0])
            >>> qc.measure()

        Example 2:
            >>> import renom_q
            >>> q = renom_q.QuantumRegister(2)
            >>> c = renom_q.ClassicalRegister(2)
            >>> qc = renom_q.QuantumCircuit(q, c)
            >>> qc.h(q[0])
            >>> qc.measure(q, c)

        Example 3:
            >>> import renom_q
            >>> q = renom_q.QuantumRegister(2)
            >>> c = renom_q.ClassicalRegister(2)
            >>> qc = renom_q.QuantumCircuit(q, c)
            >>> qc.h(q[0])
            >>> for i in range(2):
            ...     qc.measure(q[i], c[i])
        """
        if self.measure_bool:
            self.statevector.append(self.Qr.qubit)
            idx = len(self.statevector) - 1
            self.Codelist.append('list=QC.measure_exec(QC.statevector[' + str(idx) + ']);\n')
            self.measure_bool = False

        if args == ():
            c_name = self.Cr.name
            for qc in range(len(self.Qr.numlist)):
                q_name = [k for k, v in self.Qr.dict.items() if v == qc][0]
                val = self.Qr.dict[q_name]
                for i in range(self.Qr.numlist[val]):
                    q_name, calc_num1 = QuantumCircuit.convert_q_number(self, (q_name, i))
                    self.Codelist.append(
                        'QC.Cr.cbit[' + str(calc_num1) + '] = list[' + str(calc_num1) + '];\n')
                    self.qasmlist.append('measure ' + q_name + '[' + str(i) + '] -> '
                                         + c_name + '[' + str(calc_num1) + '];')

        elif type(args[0]) is QuantumRegister and type(args[1]) is ClassicalRegister:
            q_name = args[0].name
            c_name = args[1].name
            val = self.Qr.dict[q_name]
            for i in range(self.Qr.numlist[val]):
                q_name, calc_num1 = QuantumCircuit.convert_q_number(self, (q_name, i))
                c_name, calc_num2 = QuantumCircuit.convert_c_number(self, (c_name, i))
                self.Codelist.append(
                    'QC.Cr.cbit[' + str(calc_num2) + '] = list[' + str(calc_num1) + '];\n')
                self.qasmlist.append('measure ' + q_name + '[' + str(i) + '] -> '
                                     + c_name + '[' + str(i) + '];')

        elif type(args[0]) is tuple and type(args[1]) is tuple:
            q_name, num1 = args[0]
            q_name, calc_num1 = QuantumCircuit.convert_q_number(self, args[0])
            c_name, num2 = args[1]
            c_name, calc_num2 = QuantumCircuit.convert_c_number(self, args[1])
            self.Codelist.append('QC.Cr.cbit[' + str(calc_num2) +
                                 '] = list[' + str(calc_num1) + '];\n')
            self.qasmlist.append('measure ' + q_name + '[' + str(num1) + '] -> '
                                 + c_name + '[' + str(num2) + '];')

        return self

    def convert_q_number(self, q_num):
        name, num = q_num
        val = self.Qr.dict[name]
        for i in range(val):
            num += self.Qr.numlist[i]
        return name, num

    def convert_c_number(self, c_num):
        name, num = c_num
        val = self.Cr.dict[name]
        for i in range(val):
            num += self.Cr.numlist[i]
        return name, num

    def gate_base1(self, gate_matrix, num, gatemark):
        self.measure_bool = True
        mark = ''
        tensor = ''
        if num == 0 and self.Qr.num == 1:
            gate = gate_matrix
            mark = gatemark
            tensor += str(gate)
            tensor = '\n---------------- ' + mark + ' ----------------\n' + tensor
            self.tensorlist.append(tensor)

        elif num == 0 and self.Qr.num != 1:
            gate = np.kron(gate_matrix, self.idgate)
            mark += str(gatemark) + ' ⊗ I'
            tensor += str(gate_matrix) + ' ⊗ \n\n' + str(self.idgate)
            for i in range(self.Qr.num - num - 2):
                gate = np.kron(gate, self.idgate)
                mark += ' ⊗ I'
                tensor += ' ⊗ \n\n' + str(self.idgate)
            tensor += ' = \n\n' + str(gate)
            tensor = '\n---------------- ' + mark + ' ----------------\n' + tensor
            self.tensorlist.append(tensor)

        elif num == 1 and self.Qr.num == 2:
            gate = np.kron(self.idgate, gate_matrix)
            mark += 'I ⊗ ' + str(gatemark)
            tensor += str(self.idgate) + ' ⊗ \n\n' + str(gate_matrix) + ' = \n\n' + str(gate)
            tensor = '\n---------------- ' + mark + ' ----------------\n' + tensor
            self.tensorlist.append(tensor)

        elif num == 1 and self.Qr.num != 2:
            gate = np.kron(self.idgate, gate_matrix)
            mark += 'I ⊗ ' + str(gatemark)
            tensor += str(self.idgate) + ' ⊗ \n\n' + str(gate_matrix)
            for i in range(self.Qr.num - num - 1):
                gate = np.kron(gate, self.idgate)
                mark += ' ⊗ I'
                tensor += ' ⊗ \n\n' + str(self.idgate)
            tensor += ' = \n\n' + str(gate)
            tensor = '\n---------------- ' + mark + ' ----------------\n' + tensor
            self.tensorlist.append(tensor)

        elif num == self.Qr.num - 1:
            gate = np.kron(self.idgate, self.idgate)
            mark += 'I ⊗ I'
            tensor += str(self.idgate) + ' ⊗ \n\n' + str(self.idgate)
            for i in range(num - 2):
                gate = np.kron(gate, self.idgate)
                mark += ' ⊗ I'
                tensor += ' ⊗ \n\n' + str(self.idgate)
            gate = np.kron(gate, gate_matrix)
            mark += ' ⊗ ' + str(gatemark)
            tensor += ' ⊗ \n\n' + str(gate_matrix) + ' = \n\n' + str(gate)
            tensor = '\n---------------- ' + mark + ' ----------------\n' + tensor
            self.tensorlist.append(tensor)

        else:
            gate = np.kron(self.idgate, self.idgate)
            mark += 'I ⊗ I'
            tensor += str(self.idgate) + ' ⊗ \n\n' + str(self.idgate)
            for i in range(num - 2):
                gate = np.kron(gate, self.idgate)
                mark += ' ⊗ I'
                tensor += ' ⊗ \n\n' + str(self.idgate)
            gate = np.kron(gate, gate_matrix)
            mark += ' ⊗ ' + str(gatemark)
            tensor += ' ⊗ \n\n' + str(gate_matrix)
            for i in range(self.Qr.num - num - 1):
                gate = np.kron(gate, self.idgate)
                mark += ' ⊗ I'
                tensor += ' ⊗ \n\n' + str(self.idgate)
            tensor += ' = \n\n' + str(gate)
            tensor = '\n---------------- ' + mark + ' ----------------\n' + tensor
            self.tensorlist.append(tensor)

        return gate

    def reset(self):
        """ Apply reset gate to quantum register.

        Thie gate is available only when resetting all quantum register.

        Example:
            >>> import renom_q
            >>> q = renom_q.QuantumRegister(1)
            >>> c = renom_q.ClassicalRegister(1)
            >>> qc = renom_q.QuantumCircuit(q, c)
            >>> qc.reset()
        """
        qubit = self.Qr.qubit
        self.Qr.qubit = np.array([complex(0.) for i in range(self.Qr.q_states)])
        self.Qr.qubit[0] = 1.
        self.matrixlist.append('\n---------------- reset() ----------------\n'
                               + str(np.array(qubit).reshape(self.Qr.q_states, 1)) +
                               ' → \n\n' +
                               str(np.array(self.Qr.qubit).reshape(self.Qr.q_states, 1)))
        for qc in range(len(self.Qr.numlist)):
            q_name = [k for k, v in self.Qr.dict.items() if v == qc][0]
            val = self.Qr.dict[q_name]
            for i in range(self.Qr.numlist[val]):
                self.qasmlist.append('reset ' + q_name + '[' + str(i) + '];')
        return self

    def id(self, q_num):
        """ Apply id gate to quantum register.

        Args:
            q_num (tuple):
                A tuple of a quantum register and its index (ex:q[0]).

        Example:
            >>> import renom_q
            >>> q = renom_q.QuantumRegister(1)
            >>> c = renom_q.ClassicalRegister(1)
            >>> qc = renom_q.QuantumCircuit(q, c)
            >>> qc.id(q[0])
        """
        name, origin_num = q_num
        name, num = QuantumCircuit.convert_q_number(self, q_num)
        gate = QuantumCircuit.gate_base1(self, self.idgate, num, 'I')
        qubit = self.Qr.qubit
        self.Qr.qubit = np.dot(gate, self.Qr.qubit)
        self.matrixlist.append('\n---------------- id(' + name + '[' + str(origin_num) + ']) ----------------\n'
                               + str(gate) + '・\n\n' + str(np.array(qubit).reshape(self.Qr.q_states, 1)) +
                               ' = \n\n' + str(np.array(self.Qr.qubit).reshape(self.Qr.q_states, 1)))
        self.qasmlist.append('id ' + name + '[' + str(origin_num) + '];')
        return self

    def x(self, q_num):
        """ Apply x gate to quantum register.

        Args:
            q_num (tuple):
                A tuple of a quantum register and its index (ex:q[0]).

        Example:
            >>> import renom_q
            >>> q = renom_q.QuantumRegister(1)
            >>> c = renom_q.ClassicalRegister(1)
            >>> qc = renom_q.QuantumCircuit(q, c)
            >>> qc.x(q[0])
        """
        name, origin_num = q_num
        name, num = QuantumCircuit.convert_q_number(self, q_num)
        xgate = np.array([[0., 1.], [1., 0.]])
        gate = QuantumCircuit.gate_base1(self, xgate, num, 'X')
        qubit = self.Qr.qubit
        self.Qr.qubit = np.dot(gate, self.Qr.qubit)
        self.matrixlist.append('\n---------------- x(' + name + '[' + str(origin_num) + ']) ----------------\n'
                               + str(gate) + '・\n\n' + str(np.array(qubit).reshape(self.Qr.q_states, 1)) +
                               ' = \n\n' + str(np.array(self.Qr.qubit).reshape(self.Qr.q_states, 1)))
        self.qasmlist.append('x ' + name + '[' + str(origin_num) + '];')
        return self

    def z(self, q_num):
        """ Apply z gate to quantum register.

        Args:
            q_num (tuple):
                A tuple of a quantum register and its index (ex:q[0]).

        Example:
            >>> import renom_q
            >>> q = renom_q.QuantumRegister(1)
            >>> c = renom_q.ClassicalRegister(1)
            >>> qc = renom_q.QuantumCircuit(q, c)
            >>> qc.z(q[0])
        """
        name, origin_num = q_num
        name, num = QuantumCircuit.convert_q_number(self, q_num)
        zgate = np.array([[1., 0.], [0., -1.]])
        gate = QuantumCircuit.gate_base1(self, zgate, num, 'Z')
        qubit = self.Qr.qubit
        self.Qr.qubit = np.dot(gate, self.Qr.qubit)
        self.matrixlist.append('\n---------------- z(' + name + '[' + str(origin_num) + ']) ----------------\n'
                               + str(gate) + '・\n\n' + str(np.array(qubit).reshape(self.Qr.q_states, 1)) +
                               ' = \n\n' + str(np.array(self.Qr.qubit).reshape(self.Qr.q_states, 1)))
        self.qasmlist.append('z ' + name + '[' + str(origin_num) + '];')
        return self

    def y(self, q_num):
        """ Apply y gate to quantum register.

        Args:
            q_num (tuple):
                A tuple of a quantum register and its index (ex:q[0]).

        Example:
            >>> import renom_q
            >>> q = renom_q.QuantumRegister(1)
            >>> c = renom_q.ClassicalRegister(1)
            >>> qc = renom_q.QuantumCircuit(q, c)
            >>> qc.y(q[0])
        """
        name, origin_num = q_num
        name, num = QuantumCircuit.convert_q_number(self, q_num)
        ygate = np.array([[0., -1.0j], [1.0j, 0.]])
        gate = QuantumCircuit.gate_base1(self, ygate, num, 'Y')
        qubit = self.Qr.qubit
        self.Qr.qubit = np.dot(gate, self.Qr.qubit)
        self.matrixlist.append('\n---------------- y(' + name + '[' + str(origin_num) + ']) ----------------\n'
                               + str(gate) + '・\n\n' + str(np.array(qubit).reshape(self.Qr.q_states, 1)) +
                               ' = \n\n' + str(np.array(self.Qr.qubit).reshape(self.Qr.q_states, 1)))
        self.qasmlist.append('y ' + name + '[' + str(origin_num) + '];')
        return self

    def h(self, q_num):
        """ Apply h gate to quantum register.

        Args:
            q_num (tuple):
                A tuple of a quantum register and its index (ex:q[0]).

        Example:
            >>> import renom_q
            >>> q = renom_q.QuantumRegister(1)
            >>> c = renom_q.ClassicalRegister(1)
            >>> qc = renom_q.QuantumCircuit(q, c)
            >>> qc.h(q[0])
        """
        name, origin_num = q_num
        name, num = QuantumCircuit.convert_q_number(self, q_num)
        hgate = (1 / np.sqrt(2)) * np.array([[1., 1.], [1., -1.]])
        gate = QuantumCircuit.gate_base1(self, hgate, num, 'H')
        qubit = self.Qr.qubit
        self.Qr.qubit = np.dot(gate, self.Qr.qubit)
        self.matrixlist.append('\n---------------- h(' + name + '[' + str(origin_num) + ']) ----------------\n'
                               + str(gate) + '・\n\n' + str(np.array(qubit).reshape(self.Qr.q_states, 1)) +
                               ' = \n\n' + str(np.array(self.Qr.qubit).reshape(self.Qr.q_states, 1)))
        self.qasmlist.append('h ' + name + '[' + str(origin_num) + '];')
        return self

    def s(self, q_num):
        """ Apply s gate to quantum register.

        Args:
            q_num (tuple):
                A tuple of a quantum register and its index (ex:q[0]).

        Example:
            >>> import renom_q
            >>> q = renom_q.QuantumRegister(1)
            >>> c = renom_q.ClassicalRegister(1)
            >>> qc = renom_q.QuantumCircuit(q, c)
            >>> qc.s(q[0])
        """
        name, origin_num = q_num
        name, num = QuantumCircuit.convert_q_number(self, q_num)
        sgate = np.array([[1., 0.], [0., 1.0j]])
        gate = QuantumCircuit.gate_base1(self, sgate, num, 'S')
        qubit = self.Qr.qubit
        self.Qr.qubit = np.dot(gate, self.Qr.qubit)
        self.matrixlist.append('\n---------------- s(' + name + '[' + str(origin_num) + ']) ----------------\n'
                               + str(gate) + '・\n\n' + str(np.array(qubit).reshape(self.Qr.q_states, 1)) +
                               ' = \n\n' + str(np.array(self.Qr.qubit).reshape(self.Qr.q_states, 1)))
        self.qasmlist.append('s ' + name + '[' + str(origin_num) + '];')
        return self

    def sdg(self, q_num):
        """ Apply sdg gate to quantum register.

        Args:
            q_num (tuple):
                A tuple of a quantum register and its index (ex:q[0]).

        Example:
            >>> import renom_q
            >>> q = renom_q.QuantumRegister(1)
            >>> c = renom_q.ClassicalRegister(1)
            >>> qc = renom_q.QuantumCircuit(q, c)
            >>> qc.sdg(q[0])
        """
        name, origin_num = q_num
        name, num = QuantumCircuit.convert_q_number(self, q_num)
        sdggate = np.array([[1., 0.], [0., -1.0j]])
        gate = QuantumCircuit.gate_base1(self, sdggate, num, 'S^†')
        qubit = self.Qr.qubit
        self.Qr.qubit = np.dot(gate, self.Qr.qubit)
        self.matrixlist.append('\n---------------- sdg(' + name + '[' + str(origin_num) + ']) ----------------\n'
                               + str(gate) + '・\n\n' + str(np.array(qubit).reshape(self.Qr.q_states, 1)) +
                               ' = \n\n' + str(np.array(self.Qr.qubit).reshape(self.Qr.q_states, 1)))
        self.qasmlist.append('sdg ' + name + '[' + str(origin_num) + '];')
        return self

    def t(self, q_num):
        """ Apply t gate to quantum register.

        Args:
            q_num (tuple):
                A tuple of a quantum register and its index (ex:q[0]).

        Example:
            >>> import renom_q
            >>> q = renom_q.QuantumRegister(1)
            >>> c = renom_q.ClassicalRegister(1)
            >>> qc = renom_q.QuantumCircuit(q, c)
            >>> qc.t(q[0])
        """
        name, origin_num = q_num
        name, num = QuantumCircuit.convert_q_number(self, q_num)
        tgate = np.array([[1., 0.], [0., (1. + 1.0j) / np.sqrt(2)]])
        gate = QuantumCircuit.gate_base1(self, tgate, num, 'T')
        qubit = self.Qr.qubit
        self.Qr.qubit = np.dot(gate, self.Qr.qubit)
        self.matrixlist.append('\n---------------- t(' + name + '[' + str(origin_num) + ']) ----------------\n'
                               + str(gate) + '・\n\n' + str(np.array(qubit).reshape(self.Qr.q_states, 1)) +
                               ' = \n\n' + str(np.array(self.Qr.qubit).reshape(self.Qr.q_states, 1)))
        self.qasmlist.append('t ' + name + '[' + str(origin_num) + '];')
        return self

    def tdg(self, q_num):
        """ Apply tdg gate to quantum register.

        Args:
            q_num (tuple):
                A tuple of a quantum register and its index (ex:q[0]).

        Example:
            >>> import renom_q
            >>> q = renom_q.QuantumRegister(1)
            >>> c = renom_q.ClassicalRegister(1)
            >>> qc = renom_q.QuantumCircuit(q, c)
            >>> qc.tsg(q[0])
        """
        name, origin_num = q_num
        name, num = QuantumCircuit.convert_q_number(self, q_num)
        tdggate = np.array([[1., 0.], [0., (1. - 1.0j) / np.sqrt(2)]])
        gate = QuantumCircuit.gate_base1(self, tdggate, num, 'T^†')
        qubit = self.Qr.qubit
        self.Qr.qubit = np.dot(gate, self.Qr.qubit)
        self.matrixlist.append('\n---------------- tdg(' + name + '[' + str(origin_num) + ']) ----------------\n'
                               + str(gate) + '・\n\n' + str(np.array(qubit).reshape(self.Qr.q_states, 1)) +
                               ' = \n\n' + str(np.array(self.Qr.qubit).reshape(self.Qr.q_states, 1)))
        self.qasmlist.append('tdg ' + name + '[' + str(origin_num) + '];')
        return self

    def rx(self, theta, q_num):
        """ Apply rx gate to quantum register.

        Args:
            theta (float):
                Rotation angle of quantum statevector.
            q_num (tuple):
                A tuple of a quantum register and its index (ex:q[0]).

        Example:
            >>> import renom_q
            >>> q = renom_q.QuantumRegister(1)
            >>> c = renom_q.ClassicalRegister(1)
            >>> qc = renom_q.QuantumCircuit(q, c)
            >>> qc.rx(math.pi, q[0])
        """
        name, origin_num = q_num
        name, num = QuantumCircuit.convert_q_number(self, q_num)
        rxgate = np.array([[np.cos(theta / 2), -1.0j * np.sin(theta / 2)],
                           [-1.0j * np.sin(theta / 2), np.cos(theta / 2)]])
        gate = QuantumCircuit.gate_base1(self, rxgate, num, 'Rx')
        qubit = self.Qr.qubit
        self.Qr.qubit = np.dot(gate, self.Qr.qubit)
        self.matrixlist.append('\n---------------- rx(' + str(theta) + ', ' + name + '[' + str(origin_num) + ']) ----------------\n'
                               + str(gate) + '・\n\n' + str(np.array(qubit).reshape(self.Qr.q_states, 1)) +
                               ' = \n\n' + str(np.array(self.Qr.qubit).reshape(self.Qr.q_states, 1)))
        self.qasmlist.append('rx(' + str(theta) + ') ' + name + '[' + str(origin_num) + '];')
        return self

    def ry(self, theta, q_num):
        """ Apply ry gate to quantum register.

        Args:
            theta (float):
                Rotation angle of quantum statevector.
            q_num (tuple):
                A tuple of a quantum register and its index (ex:q[0]).

        Example:
            >>> import renom_q
            >>> q = renom_q.QuantumRegister(1)
            >>> c = renom_q.ClassicalRegister(1)
            >>> qc = renom_q.QuantumCircuit(q, c)
            >>> qc.ry(math.pi, q[0])
        """
        name, origin_num = q_num
        name, num = QuantumCircuit.convert_q_number(self, q_num)
        rygate = np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                           [np.sin(theta / 2), np.cos(theta / 2)]])
        gate = QuantumCircuit.gate_base1(self, rygate, num, 'Ry')
        qubit = self.Qr.qubit
        self.Qr.qubit = np.dot(gate, self.Qr.qubit)
        self.matrixlist.append('\n---------------- ry(' + str(theta) + ', ' + name + '[' + str(origin_num) + ']) ----------------\n'
                               + str(gate) + '・\n\n' + str(np.array(qubit).reshape(self.Qr.q_states, 1)) +
                               ' = \n\n' + str(np.array(self.Qr.qubit).reshape(self.Qr.q_states, 1)))
        self.qasmlist.append('ry(' + str(theta) + ') ' + name + '[' + str(origin_num) + '];')
        return self

    def rz(self, theta, q_num):
        """ Apply rz gate to quantum register.

        Args:
            theta (float):
                Rotation angle of quantum statevector.
            q_num (tuple):
                A tuple of a quantum register and its index (ex:q[0]).

        Example:
            >>> import renom_q
            >>> q = renom_q.QuantumRegister(1)
            >>> c = renom_q.ClassicalRegister(1)
            >>> qc = renom_q.QuantumCircuit(q, c)
            >>> qc.rz(math.pi, q[0])
        """
        name, origin_num = q_num
        name, num = QuantumCircuit.convert_q_number(self, q_num)
        rzgate = np.array([[1., 0.], [0., np.exp(1.0j * theta)]])
        gate = QuantumCircuit.gate_base1(self, rzgate, num, 'Rz')
        qubit = self.Qr.qubit
        self.Qr.qubit = np.dot(gate, self.Qr.qubit)
        self.matrixlist.append('\n---------------- rz(' + str(theta) + ', ' + name + '[' + str(origin_num) + ']) ----------------\n'
                               + str(gate) + '・\n\n' + str(np.array(qubit).reshape(self.Qr.q_states, 1)) +
                               ' = \n\n' + str(np.array(self.Qr.qubit).reshape(self.Qr.q_states, 1)))
        self.qasmlist.append('rz(' + str(theta) + ') ' + name + '[' + str(origin_num) + '];')
        return self

    def u1(self, lam, q_num):
        """ Apply u1 gate to quantum register.

        Args:
            lam (float):
                The paramater of unitary gate U1.
            q_num (tuple):
                A tuple of a quantum register and its index (ex:q[0]).

        Example:
            >>> import renom_q
            >>> q = renom_q.QuantumRegister(1)
            >>> c = renom_q.ClassicalRegister(1)
            >>> qc = renom_q.QuantumCircuit(q, c)
            >>> qc.u1(math.pi, q[0])
        """
        name, origin_num = q_num
        name, num = QuantumCircuit.convert_q_number(self, q_num)
        u1gate = np.array([[1., 0.], [0., np.exp(1.0j * lam)]])
        gate = QuantumCircuit.gate_base1(self, u1gate, num, 'U1')
        qubit = self.Qr.qubit
        self.Qr.qubit = np.dot(gate, self.Qr.qubit)
        self.matrixlist.append('\n---------------- u1(' + str(lam) + ', ' + name + '[' + str(origin_num) + ']) ----------------\n'
                               + str(gate) + '・\n\n' + str(np.array(qubit).reshape(self.Qr.q_states, 1)) +
                               ' = \n\n' + str(np.array(self.Qr.qubit).reshape(self.Qr.q_states, 1)))
        self.qasmlist.append('u1(' + str(lam) + ') ' + name + '[' + str(origin_num) + '];')
        return self

    def u2(self, phi, lam, q_num):
        """ Apply u2 gate to quantum register.

        Args:
            phi (float):
                The paramater of unitary gate U2.
            lam (float):
                The paramater of unitary gate U2.
            q_num (tuple):
                A tuple of a quantum register and its index (ex:q[0]).

        Example:
            >>> import renom_q
            >>> q = renom_q.QuantumRegister(1)
            >>> c = renom_q.ClassicalRegister(1)
            >>> qc = renom_q.QuantumCircuit(q, c)
            >>> qc.u2(0, math.pi, q[0])
        """
        name, origin_num = q_num
        name, num = QuantumCircuit.convert_q_number(self, q_num)
        u2gate = (1 / np.sqrt(2)) * np.array([[1., -np.exp(1.0j * lam)],
                                              [np.exp(1.0j * phi), np.exp(1.0j * (lam + phi))]])
        gate = QuantumCircuit.gate_base1(self, u2gate, num, 'U2')
        qubit = self.Qr.qubit
        self.Qr.qubit = np.dot(gate, self.Qr.qubit)
        self.matrixlist.append('\n---------------- u2(' + str(phi) + ', ' + str(lam) +
                               ', ' + name + '[' + str(origin_num) + ']) ----------------\n'
                               + str(gate) + '・\n\n' + str(np.array(qubit).reshape(self.Qr.q_states, 1)) +
                               ' = \n\n' + str(np.array(self.Qr.qubit).reshape(self.Qr.q_states, 1)))
        self.qasmlist.append('u2(' + str(phi) + ', ' + str(lam) + ') '
                             + name + '[' + str(origin_num) + '];')
        return self

    def u3(self, theta, phi, lam, q_num):
        """ Apply u3 gate to quantum register.

        Args:
            theta (float):
                The paramater of unitary gate U3.
            phi (float):
                The paramater of unitary gate U3.
            lam (float):
                The paramater of unitary gate U3.
            q_num (tuple):
                A tuple of a quantum register and its index (ex:q[0]).

        Example:
            >>> import renom_q
            >>> q = renom_q.QuantumRegister(1)
            >>> c = renom_q.ClassicalRegister(1)
            >>> qc = renom_q.QuantumCircuit(q, c)
            >>> qc.u3(math.pi, 0, math.pi, q[0])
        """
        name, origin_num = q_num
        name, num = QuantumCircuit.convert_q_number(self, q_num)
        u3gate = np.array([[np.cos(theta / 2), -np.exp(1.0j * lam) * np.sin(theta / 2)],
                           [np.exp(1.0j * phi) * np.sin(theta / 2), np.exp(1.0j * (lam + phi)) * np.cos(theta / 2)]])
        gate = QuantumCircuit.gate_base1(self, u3gate, num, 'U3')
        qubit = self.Qr.qubit
        self.Qr.qubit = np.dot(gate, self.Qr.qubit)
        self.matrixlist.append('\n---------------- u3(' + str(theta) + ', ' + str(phi) +
                               ', ' + str(lam) + ', ' + name +
                               '[' + str(origin_num) + ']) ----------------\n'
                               + str(gate) + '・\n\n' + str(np.array(qubit).reshape(self.Qr.q_states, 1)) +
                               ' = \n\n' + str(np.array(self.Qr.qubit).reshape(self.Qr.q_states, 1)))
        self.qasmlist.append('u3(' + str(theta) + ', ' + str(phi) + ', ' + str(lam) +
                             ') ' + name + '[' + str(origin_num) + '];')
        return self

    def gate_base2(self, gate_matrix, ctl, tgt, gatemark):
        self.measure_bool = True
        gate = np.zeros((self.Qr.q_states, self.Qr.q_states), dtype=complex)
        for i in range(self.Qr.q_states):
            bit_c = int(int(format(i, 'b')) / 10**(self.Qr.num - ctl - 1) % 2)
            if bit_c == 1:
                bit_t = int(int(format(i, 'b')) / 10**(self.Qr.num - tgt - 1) % 2)
                bit_list = list(format(i, '0' + str(self.Qr.num) + 'b'))
                bit_list[tgt] = '1' if bit_t == 0 else '0'
                idx = int("".join(bit_list), 2)
                if i < idx:
                    gate[i, i] = gate_matrix[bit_t, 0]
                    gate[i, idx] = gate_matrix[bit_t, 1]
                else:
                    gate[i, i] = gate_matrix[bit_t, 1]
                    gate[i, idx] = gate_matrix[bit_t, 0]
            else:
                gate[i, i] = 1.
        tensor = '\n---------------- ' + gatemark + ' ----------------\n' + str(gate)
        self.tensorlist.append(tensor)
        return gate

    def cx(self, q_num1, q_num2):
        """ Apply cx gate to quantum register.

        Args:
            q_num1 (tuple):
                A tuple of a quantum register and its index (ex:q[0]). It's the
                control qubit.
            q_num2 (tuple):
                A tuple of a quantum register and its index (ex:q[0]). It's the
                target qubit.

        Example:
            >>> import renom_q
            >>> q = renom_q.QuantumRegister(2)
            >>> c = renom_q.ClassicalRegister(2)
            >>> qc = renom_q.QuantumCircuit(q, c)
            >>> qc.cx(q[0], q[1])
        """
        name1, origin_num1 = q_num1
        name1, num1 = QuantumCircuit.convert_q_number(self, q_num1)
        name2, origin_num2 = q_num2
        name2, num2 = QuantumCircuit.convert_q_number(self, q_num2)
        xgate = np.array([[0., 1.], [1., 0.]])
        gate = QuantumCircuit.gate_base2(self, xgate, num1, num2, 'cX')
        qubit = self.Qr.qubit
        self.Qr.qubit = np.dot(gate, self.Qr.qubit)
        self.matrixlist.append('\n---------------- cx(' + name1 + '[' + str(origin_num1) + '], '
                               + name2 + '[' + str(origin_num2) + ']) ----------------\n'
                               + str(gate) + '・\n\n' + str(np.array(qubit).reshape(self.Qr.q_states, 1)) +
                               ' = \n\n' + str(np.array(self.Qr.qubit).reshape(self.Qr.q_states, 1)))
        self.qasmlist.append('cx ' + name1 + '[' + str(origin_num1) + '], '
                             + name2 + '[' + str(origin_num2) + '];')
        return self

    def cy(self, q_num1, q_num2):
        """ Apply cy gate to quantum register.

        Args:
            q_num1 (tuple):
                A tuple of a quantum register and its index (ex:q[0]). It's the
                control qubit.
            q_num2 (tuple):
                A tuple of a quantum register and its index (ex:q[0]). It's the
                target qubit.

        Example:
            >>> import renom_q
            >>> q = renom_q.QuantumRegister(2)
            >>> c = renom_q.ClassicalRegister(2)
            >>> qc = renom_q.QuantumCircuit(q, c)
            >>> qc.cy(q[0], q[1])
        """
        name1, origin_num1 = q_num1
        name1, num1 = QuantumCircuit.convert_q_number(self, q_num1)
        name2, origin_num2 = q_num2
        name2, num2 = QuantumCircuit.convert_q_number(self, q_num2)
        ygate = np.array([[0., -1.0j], [1.0j, 0.]])
        gate = QuantumCircuit.gate_base2(self, ygate, num1, num2, 'cY')
        qubit = self.Qr.qubit
        self.Qr.qubit = np.dot(gate, self.Qr.qubit)
        self.matrixlist.append('\n---------------- cy(' + name1 + '[' + str(origin_num1) +
                               '], ' + name2 + '[' + str(origin_num2) + ']) ----------------\n'
                               + str(gate) + '・\n\n' + str(np.array(qubit).reshape(self.Qr.q_states, 1)) +
                               ' = \n\n' + str(np.array(self.Qr.qubit).reshape(self.Qr.q_states, 1)))
        self.qasmlist.append('cy ' + name1 + '[' + str(origin_num1) + '], '
                             + name2 + '[' + str(origin_num2) + '];')
        return self

    def cz(self, q_num1, q_num2):
        """ Apply cz gate to quantum register.

        Args:
            q_num1 (tuple):
                A tuple of a quantum register and its index (ex:q[0]). It's the
                control qubit.
            q_num2 (tuple):
                A tuple of a quantum register and its index (ex:q[0]). It's the
                target qubit.

        Example:
            >>> import renom_q
            >>> q = renom_q.QuantumRegister(2)
            >>> c = renom_q.ClassicalRegister(2)
            >>> qc = renom_q.QuantumCircuit(q, c)
            >>> qc.cz(q[0], q[1])
        """
        name1, origin_num1 = q_num1
        name1, num1 = QuantumCircuit.convert_q_number(self, q_num1)
        name2, origin_num2 = q_num2
        name2, num2 = QuantumCircuit.convert_q_number(self, q_num2)
        zgate = np.array([[1., 0.], [0., -1.]])
        gate = QuantumCircuit.gate_base2(self, zgate, num1, num2, 'cZ')
        qubit = self.Qr.qubit
        self.Qr.qubit = np.dot(gate, self.Qr.qubit)
        self.matrixlist.append('\n---------------- cz(' + name1 + '[' + str(origin_num1) +
                               '], ' + name2 + '[' + str(origin_num2) + ']) ----------------\n'
                               + str(gate) + '・\n\n' + str(np.array(qubit).reshape(self.Qr.q_states, 1)) +
                               ' = \n\n' + str(np.array(self.Qr.qubit).reshape(self.Qr.q_states, 1)))
        self.qasmlist.append('cz ' + name1 + '[' + str(origin_num1) + '], '
                             + name2 + '[' + str(origin_num2) + '];')
        return self

    def ch(self, q_num1, q_num2):
        """ Apply ch gate to quantum register.

        Args:
            q_num1 (tuple):
                A tuple of a quantum register and its index (ex:q[0]). It's the
                control qubit.
            q_num2 (tuple):
                A tuple of a quantum register and its index (ex:q[0]). It's the
                target qubit.

        Example:
            >>> import renom_q
            >>> q = renom_q.QuantumRegister(2)
            >>> c = renom_q.ClassicalRegister(2)
            >>> qc = renom_q.QuantumCircuit(q, c)
            >>> qc.ch(q[0], q[1])
        """
        name1, origin_num1 = q_num1
        name1, num1 = QuantumCircuit.convert_q_number(self, q_num1)
        name2, origin_num2 = q_num2
        name2, num2 = QuantumCircuit.convert_q_number(self, q_num2)
        hgate = (1 / np.sqrt(2)) * np.array([[1., 1.], [1., -1.]])
        gate = QuantumCircuit.gate_base2(self, hgate, num1, num2, 'cH')
        qubit = self.Qr.qubit
        self.Qr.qubit = np.dot(gate, self.Qr.qubit)
        self.matrixlist.append('\n---------------- ch(' + name1 + '[' + str(origin_num1) +
                               '], ' + name2 + '[' + str(origin_num2) + ']) ----------------\n'
                               + str(gate) + '・\n\n' + str(np.array(qubit).reshape(self.Qr.q_states, 1)) +
                               ' = \n\n' + str(np.array(self.Qr.qubit).reshape(self.Qr.q_states, 1)))
        self.qasmlist.append('ch ' + name1 + '[' + str(origin_num1) + '], '
                             + name2 + '[' + str(origin_num2) + '];')
        return self

    def cs(self, q_num1, q_num2):
        """ Apply cs gate to quantum register.

        Cannot draw the cs gate in QuantumCircuit.draw_circuit().

        Args:
            q_num1 (tuple):
                A tuple of a quantum register and its index (ex:q[0]). It's the
                control qubit.
            q_num2 (tuple):
                A tuple of a quantum register and its index (ex:q[0]). It's the
                target qubit.

        Example:
            >>> import renom_q
            >>> q = renom_q.QuantumRegister(2)
            >>> c = renom_q.ClassicalRegister(2)
            >>> qc = renom_q.QuantumCircuit(q, c)
            >>> qc.cs(q[0], q[1])
        """
        name1, origin_num1 = q_num1
        name1, num1 = QuantumCircuit.convert_q_number(self, q_num1)
        name2, origin_num2 = q_num2
        name2, num2 = QuantumCircuit.convert_q_number(self, q_num2)
        sgate = np.array([[1., 0.], [0., 1.0j]])
        gate = QuantumCircuit.gate_base2(self, sgate, num1, num2, 'cS')
        qubit = self.Qr.qubit
        self.Qr.qubit = np.dot(gate, self.Qr.qubit)
        self.matrixlist.append('\n---------------- cs(' + name1 + '[' + str(origin_num1) +
                               '], ' + name2 + '[' + str(origin_num2) + ']) ----------------\n'
                               + str(gate) + '・\n\n' + str(np.array(qubit).reshape(self.Qr.q_states, 1)) +
                               ' = \n\n' + str(np.array(self.Qr.qubit).reshape(self.Qr.q_states, 1)))
        return self

    def cu1(self, lam, q_num1, q_num2):
        """ Apply cu1 gate to quantum register.

        Args:
            lam (float):
                The paramater of unitary gate cU1.
            q_num1 (tuple):
                A tuple of a quantum register and its index (ex:q[0]). It's the
                control qubit.
            q_num2 (tuple):
                A tuple of a quantum register and its index (ex:q[0]). It's the
                target qubit.

        Example:
            >>> import renom_q
            >>> q = renom_q.QuantumRegister(2)
            >>> c = renom_q.ClassicalRegister(2)
            >>> qc = renom_q.QuantumCircuit(q, c)
            >>> qc.cu1(math.pi, q[0], q[1])
        """
        name1, origin_num1 = q_num1
        name1, num1 = QuantumCircuit.convert_q_number(self, q_num1)
        name2, origin_num2 = q_num2
        name2, num2 = QuantumCircuit.convert_q_number(self, q_num2)
        u1gate = np.array([[1., 0.], [0., np.exp(1.0j * lam)]])
        gate = QuantumCircuit.gate_base2(self, u1gate, num1, num2, 'cU1')
        qubit = self.Qr.qubit
        self.Qr.qubit = np.dot(gate, self.Qr.qubit)
        self.matrixlist.append('\n---------------- cu1(' + str(lam) + ', ' + name1 + '['
                               + str(origin_num1) + '], ' + name2 +
                               '[' + str(origin_num2) + ']) ----------------\n'
                               + str(gate) + '・\n\n' + str(np.array(qubit).reshape(self.Qr.q_states, 1)) +
                               ' = \n\n' + str(np.array(self.Qr.qubit).reshape(self.Qr.q_states, 1)))
        self.qasmlist.append('cu1(' + str(lam) + ') ' + name1 + '[' + str(origin_num1) + '], '
                             + name2 + '[' + str(origin_num2) + '];')
        return self

    def cu3(self, theta, phi, lam, q_num1, q_num2):
        """ Apply cu3 gate to quantum register.

        Args:
            theta (float):
                The paramater of unitary gate cU3.
            phi (float):
                The paramater of unitary gate cU3.
            lam (float):
                The paramater of unitary gate cU3.
            q_num1 (tuple):
                A tuple of a quantum register and its index (ex:q[0]). It's the
                control qubit.
            q_num2 (tuple):
                A tuple of a quantum register and its index (ex:q[0]). It's the
                target qubit.

        Example:
            >>> import renom_q
            >>> q = renom_q.QuantumRegister(2)
            >>> c = renom_q.ClassicalRegister(2)
            >>> qc = renom_q.QuantumCircuit(q, c)
            >>> qc.cu3(math.pi, 0, math.pi, q[0], q[1])
        """
        name1, origin_num1 = q_num1
        name1, num1 = QuantumCircuit.convert_q_number(self, q_num1)
        name2, origin_num2 = q_num2
        name2, num2 = QuantumCircuit.convert_q_number(self, q_num2)
        u3gate = np.array([[np.cos(theta / 2), -np.exp(1.0j * lam) * np.sin(theta / 2)],
                           [np.exp(1.0j * phi) * np.sin(theta / 2), np.exp(1.0j * (lam + phi)) * np.cos(theta / 2)]])
        gate = QuantumCircuit.gate_base2(self, u3gate, num1, num2, 'cU3')
        qubit = self.Qr.qubit
        self.Qr.qubit = np.dot(gate, self.Qr.qubit)
        self.matrixlist.append('\n---------------- cu3(' + str(theta) + ', ' + str(phi) +
                               ', ' + str(lam) + ', ' + name1 + '[' + str(origin_num1) + '], '
                               + name2 + '[' + str(origin_num2) + ']) ----------------\n'
                               + str(gate) + '・\n\n' + str(np.array(qubit).reshape(self.Qr.q_states, 1)) +
                               ' = \n\n' + str(np.array(self.Qr.qubit).reshape(self.Qr.q_states, 1)))
        self.qasmlist.append('cu3(' + str(theta) + ', ' + str(phi) + ', ' + str(lam) + ') '
                             + name1 + '[' + str(origin_num1) + '], ' + name2 + '[' + str(origin_num2) + '];')
        return self

    def crz(self, lam, q_num1, q_num2):
        """ Apply crz gate to quantum register.

        Args:
            lam (float):
                Rotation angle of quantum statevector.
            q_num1 (tuple):
                A tuple of a quantum register and its index (ex:q[0]). It's the
                control qubit.
            q_num2 (tuple):
                A tuple of a quantum register and its index (ex:q[0]). It's the
                target qubit.

        Example:
            >>> import renom_q
            >>> q = renom_q.QuantumRegister(2)
            >>> c = renom_q.ClassicalRegister(2)
            >>> qc = renom_q.QuantumCircuit(q, c)
            >>> qc.crz(math.pi, q[0], q[1])
        """
        name1, origin_num1 = q_num1
        name1, num1 = QuantumCircuit.convert_q_number(self, q_num1)
        name2, origin_num2 = q_num2
        name2, num2 = QuantumCircuit.convert_q_number(self, q_num2)
        rzgate = np.array([[1., 0.], [0., np.exp(1.0j * lam)]])
        gate = QuantumCircuit.gate_base2(self, rzgate, num1, num2, 'cRz')
        qubit = self.Qr.qubit
        self.Qr.qubit = np.dot(gate, self.Qr.qubit)
        self.matrixlist.append('\n---------------- crz(' + str(lam) + ', ' + name1 + '['
                               + str(origin_num1) + '], ' + name2 +
                               '[' + str(origin_num2) + ']) ----------------\n'
                               + str(gate) + '・\n\n' + str(np.array(qubit).reshape(self.Qr.q_states, 1)) +
                               ' = \n\n' + str(np.array(self.Qr.qubit).reshape(self.Qr.q_states, 1)))
        self.qasmlist.append('crz(' + str(lam) + ') ' + name1 + '[' + str(origin_num1) + '], '
                             + name2 + '[' + str(origin_num2) + '];')
        return self

    def swap(self, q_num1, q_num2):
        """ Apply swap gate to quantum register.

        Args:
            q_num1 (tuple):
                A tuple of a quantum register and its index (ex:q[0]). It's the
                exchanging qubit.
            q_num2 (tuple):
                A tuple of a quantum register and its index (ex:q[0]). It's the
                exchanging qubit.

        Example:
            >>> import renom_q
            >>> q = renom_q.QuantumRegister(2)
            >>> c = renom_q.ClassicalRegister(2)
            >>> qc = renom_q.QuantumCircuit(q, c)
            >>> qc.swap(q[0], q[1])
        """
        name1, origin_num1 = q_num1
        name1, num1 = QuantumCircuit.convert_q_number(self, q_num1)
        name2, origin_num2 = q_num2
        name2, num2 = QuantumCircuit.convert_q_number(self, q_num2)
        gate = np.zeros((self.Qr.q_states, self.Qr.q_states), dtype=complex)
        for i in range(self.Qr.q_states):
            bit_1 = int(int(format(i, 'b')) / 10**(self.Qr.num - num1 - 1) % 2)
            bit_2 = int(int(format(i, 'b')) / 10**(self.Qr.num - num2 - 1) % 2)
            if bit_1 == bit_2:
                gate[i, i] = 1.
            else:
                bit_list = list(format(i, '0' + str(self.Qr.num) + 'b'))
                bit_list[num1] = str(bit_2)
                bit_list[num2] = str(bit_1)
                idx = int("".join(bit_list), 2)
                gate[i, idx] = 1.
        qubit = self.Qr.qubit
        self.Qr.qubit = np.dot(gate, self.Qr.qubit)
        self.matrixlist.append('\n---------------- swap(' + name1 + '['
                               + str(origin_num1) + '], ' + name2 +
                               '[' + str(origin_num2) + ']) ----------------\n'
                               + str(gate) + '・\n\n' + str(np.array(qubit).reshape(self.Qr.q_states, 1)) +
                               ' = \n\n' + str(np.array(self.Qr.qubit).reshape(self.Qr.q_states, 1)))
        self.qasmlist.append('swap ' + name1 + '[' + str(origin_num1) + '], '
                             + name2 + '[' + str(origin_num2) + '];')
        tensor = '\n---------------- SWAP ----------------\n' + str(gate)
        self.tensorlist.append(tensor)
        return self

    def ccx(self, q_ctl1, q_ctl2, q_tgt):
        """ Apply ccx gate to quantum register.

        Args:
            q_ctl1 (tuple):
                A tuple of a quantum register and its index (ex:q[0]). It's one
                of the control qubit.
            q_ctl2 (tuple):
                A tuple of a quantum register and its index (ex:q[0]). It's one
                of the control qubit.
            q_tgt (tuple):
                A tuple of a quantum register and its index (ex:q[0]). It's the
                target qubit.

        Example:
            >>> import renom_q
            >>> q = renom_q.QuantumRegister(3)
            >>> c = renom_q.ClassicalRegister(3)
            >>> qc = renom_q.QuantumCircuit(q, c)
            >>> qc.ccx(q[0], q[1], q[2])
        """
        name_ctl1, origin_ctl1 = q_ctl1
        name_ctl1, ctl1 = QuantumCircuit.convert_q_number(self, q_ctl1)
        name_ctl2, origin_ctl2 = q_ctl2
        name_ctl2, ctl2 = QuantumCircuit.convert_q_number(self, q_ctl2)
        name_tgt, origin_tgt = q_tgt
        name_tgt, tgt = QuantumCircuit.convert_q_number(self, q_tgt)
        xgate = np.array([[0., 1.], [1., 0.]])
        gate = np.zeros((self.Qr.q_states, self.Qr.q_states), dtype=complex)
        for i in range(self.Qr.q_states):
            bit_c1 = int(int(format(i, 'b')) / 10**(self.Qr.num - ctl1 - 1) % 2)
            bit_c2 = int(int(format(i, 'b')) / 10**(self.Qr.num - ctl2 - 1) % 2)
            if bit_c1 == 1 and bit_c2 == 1:
                bit_t = int(int(format(i, 'b')) / 10**(self.Qr.num - tgt - 1) % 2)
                bit_list = list(format(i, '0' + str(self.Qr.num) + 'b'))
                bit_list[tgt] = '1' if bit_t == 0 else '0'
                idx = int("".join(bit_list), 2)
                if i < idx:
                    gate[i, i] = xgate[bit_t, 0]
                    gate[i, idx] = xgate[bit_t, 1]
                else:
                    gate[i, i] = xgate[bit_t, 1]
                    gate[i, idx] = xgate[bit_t, 0]
            else:
                gate[i, i] = 1.
        qubit = self.Qr.qubit
        self.Qr.qubit = np.dot(gate, self.Qr.qubit)
        self.matrixlist.append('\n---------------- ccx(' + name_ctl1 + '[' + str(origin_ctl1)
                               + '], ' + name_ctl2 +
                               '[' + str(origin_ctl2) + '], ' + name_tgt +
                               '[' + str(origin_tgt) + ']) ----------------\n'
                               + str(gate) + '・\n\n' + str(np.array(qubit).reshape(self.Qr.q_states, 1)) +
                               ' = \n\n' + str(np.array(self.Qr.qubit).reshape(self.Qr.q_states, 1)))
        self.qasmlist.append('ccx ' + name_ctl1 + '[' + str(origin_ctl1) + '], '
                             + name_ctl2 + '[' + str(origin_ctl2) + '], ' + name_tgt + '[' + str(origin_tgt) + '];')
        tensor = '\n---------------- ccX ----------------\n' + str(gate)
        self.tensorlist.append(tensor)
        return self

    def cswap(self, q_ctl, q_num1, q_num2):
        # q_ctl : Qubit名と制御bit, q_num1,q_num2 : Qubit名と交換するbit
        """ Apply cswap gate to quantum register.

        Args:
            q_ctl (tuple):
                A tuple of a quantum register and its index (ex:q[0]). It's the
                control qubit.
            q_num1 (tuple):
                A tuple of a quantum register and its index (ex:q[0]). It's the
                exchanging qubit.
            q_num2 (tuple):
                A tuple of a quantum register and its index (ex:q[0]). It's the
                exchanging qubit.

        Example:
            >>> import renom_q
            >>> q = renom_q.QuantumRegister(3)
            >>> c = renom_q.ClassicalRegister(3)
            >>> qc = renom_q.QuantumCircuit(q, c)
            >>> qc.cswap(q[0], q[1], q[2])
        """
        name_ctl, origin_ctl = q_ctl
        name_ctl, ctl = QuantumCircuit.convert_q_number(self, q_ctl)
        name1, origin_num1 = q_num1
        name1, num1 = QuantumCircuit.convert_q_number(self, q_num1)
        name2, origin_num2 = q_num2
        name2, num2 = QuantumCircuit.convert_q_number(self, q_num2)
        gate = np.zeros((self.Qr.q_states, self.Qr.q_states), dtype=complex)
        for i in range(self.Qr.q_states):
            bit_c = int(int(format(i, 'b')) / 10**(self.Qr.num - ctl - 1) % 2)
            if bit_c == 1:
                bit_1 = int(int(format(i, 'b')) / 10**(self.Qr.num - num1 - 1) % 2)
                bit_2 = int(int(format(i, 'b')) / 10**(self.Qr.num - num2 - 1) % 2)
                if bit_1 == bit_2:
                    gate[i, i] = 1.
                else:
                    bit_list = list(format(i, '0' + str(self.Qr.num) + 'b'))
                    bit_list[num1] = str(bit_2)
                    bit_list[num2] = str(bit_1)
                    idx = int("".join(bit_list), 2)
                    gate[i, idx] = 1.
            else:
                gate[i, i] = 1.
        qubit = self.Qr.qubit
        self.Qr.qubit = np.dot(gate, self.Qr.qubit)
        self.matrixlist.append('\n---------------- cswap(' + name_ctl + '[' + str(origin_ctl)
                               + '], ' + name1 +
                               '[' + str(origin_num1) + '], ' + name2 +
                               '[' + str(origin_num2) + ']) ----------------\n'
                               + str(gate) + '・\n\n' + str(np.array(qubit).reshape(self.Qr.q_states, 1)) +
                               ' = \n\n' + str(np.array(self.Qr.qubit).reshape(self.Qr.q_states, 1)))
        self.qasmlist.append('cswap ' + name_ctl + '[' + str(origin_ctl) + '], '
                             + name1 + '[' + str(origin_num1) + '], ' + name2 + '[' + str(origin_num2) + '];')
        tensor = '\n---------------- cSWAP ----------------\n' + str(gate)
        self.tensorlist.append(tensor)
        return self
