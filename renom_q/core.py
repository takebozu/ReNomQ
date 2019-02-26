# -*- coding: utf-8 -*-

import numpy as np
import collections
import copy
import matplotlib.pyplot as plt
from .visualization._circuit_visualization import _matplotlib_circuit_drawer as drawer_mpl
from .visualization._circuit_visualization import _text_circuit_drawer as drawer_text
from .visualization._circuit_visualization import _latex_circuit_drawer as drawer_latext
from .visualization._circuit_visualization import circuit_drawer
from .circuit.quantumcircuit import QuantumCircuit as q_ctl


class QuantumRegister:
    """definite a quantum register"""
    def __init__(self, num, name=None):
        """
        Args:
            num (int): The number of quantum bits.
            name (str): A name of quantum bits. Defaults to 'q'.

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
            num (int): The quantum bit number. If definite 3 quantum bits,
                most significant bit number is 0 and least significant bit
                number is 2.

        Returns:
            (taple): A taple of a name of quantum bits and the quantum bit
                number.

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
    """definite a classical register"""
    def __init__(self, num, name=None):
        """
        Args:
            num (int): The number of classical bits.
            name (str): A name of classical bits. Defaults to 'c'.

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
            num (int): The classical bit number. If definite 3 classical bits,
                most significant bit numberis 0 and least significant bit
                number is 2. 

        Returns:
            (taple): A taple of a name of classical bits and the classical bit
                number.

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
            (str): A name of classical bits.

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
    """
    Args:
        Counts (dict): A dict of the execution result of quantum circuit mesurement.

    Returns:
        matplotlib.figure: A matplotlib figure object for the execution result
            of quantum circuit mesurement.

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
    """
    Args:
        QC (renom_q.QuantumCircuit): A class of QuantumCircuit.
        shots (int): The number of excutions of quantum circuit mesurement.
            Defaults to 1024.

    Returns:
        (dict): A execution result of quantum circuit mesurement. The key is the
            measured classical bits. The value is the number of classical bits measured.

    Example:
        >>> import renom_q
        >>> q = renom_q.QuantumRegister(1)
        >>> c = renom_q.ClassicalRegister(1)
        >>> qc = renom_q.QuantumCircuit(q, c)
        >>> qc.measure()
        >>> renom_q.execute(qc)
        {'0': 1024}
    """
    Code = 'list=QC.measure_exec();\n'
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
    """
    Args:
        QC (renom_q.QuantumCircuit): A class of QuantumCircuit.
        tensorgate (bool): When set to True, added matrix calculation of
            quantum gate tensor product. Defaults to False.

    Returns:
        matrix(str): Strings of the final result of qubit statevector and
            all matrix calculation of unitary conversion.

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

        [[1.+0.j]
        [0.+0.j]] =

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
    """
    Args:
        QC (renom_q.QuantumCircuit): A class of QuantumCircuit.
        style (dict or str): dictionary of style or file name of style file.
            Defaults to None.

            The style dict kwarg contains numerous options that define the style of the
            output circuit visualization. While the style dict is used by the `mpl`,
            `latex`, and `latex_source` outputs some options in that are only used
            by the `mpl` output. These options are defined below, if it is only used by
            the `mpl` output it is marked as such:

            textcolor (str): The color code to use for text. Defaults to
                `'#000000'` (`mpl` only)
            subtextcolor (str): The color code to use for subtext. Defaults to
                `'#000000'` (`mpl` only)
            linecolor (str): The color code to use for lines. Defaults to
                `'#000000'` (`mpl` only)
            creglinecolor (str): The color code to use for classical register lines
                `'#778899'`(`mpl` only)
            gatetextcolor (str): The color code to use for gate text `'#000000'`
                (`mpl` only)
            gatefacecolor (str): The color code to use for gates. Defaults to
                `'#ffffff'` (`mpl` only)
            barrierfacecolor (str): The color code to use for barriers. Defaults to
                `'#bdbdbd'` (`mpl` only)
            backgroundcolor (str): The color code to use for the background.
                Defaults to `'#ffffff'` (`mpl` only)
            fontsize (int): The font size to use for text. Defaults to 13 (`mpl`
                only)
            subfontsize (int): The font size to use for subtext. Defaults to 8
                (`mpl` only)
            displaytext (dict): A dictionary of the text to use for each element
                type in the output visualization. The default values are:
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
            displaycolor (dict): The color codes to use for each circuit element.
                By default all values default to the value of `gatefacecolor` and
                the keys are the same as `displaytext`. Also, just like
                `displaytext` there is no provision for an incomplete dict passed
                in. (`mpl` only)
            latexdrawerstyle (bool): When set to True enable latex mode which will
                draw gates like the `latex` output modes. (`mpl` only)
            usepiformat (bool): When set to True use radians for output (`mpl`
                only)
            fold (int): The number of circuit elements to fold the circuit at.
                Defaults to 20 (`mpl` only)
            cregbundle (bool): If set True bundle classical registers (`mpl` only)
            showindex (bool): If set True draw an index. (`mpl` only)
            compress (bool): If set True draw a compressed circuit (`mpl` only)
            figwidth (int): The maximum width (in inches) for the output figure.
                (`mpl` only)
            dpi (int): The DPI to use for the output image. Defaults to 150 (`mpl`
                only)
            margin (list): `mpl` only
            creglinestyle (str): The style of line to use for classical registers.
                Choices are `'solid'`, `'doublet'`, or any valid matplotlib
                `linestyle` kwarg value. Defaults to `doublet`(`mpl` only)

    Returns:
        matplotlib.figure: A matplotlib figure object for the circuit diagram.

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
             'latexdrawerstyle': False, 'compress': True, 'usepiformat': True}
    input_style = default_style if style is None else style
    figure = drawer_mpl(qasm, style=input_style)
    return figure


def statevector(QC):
    """
    Args:
        QC (renom_q.QuantumCircuit): A class of QuantumCircuit.

    Returns:
        (array): A array of the qubit statevector.

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
    def __init__(self, *args, circuit_number=None):
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
                    assert qr[i].name not in q_dict, qr[i].name + " is already used. Please use a different name."
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
                    assert cr[i].name not in c_dict, cr[i].name + " is already used. Please use a different name."
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
        self.circuit_number = 0 if circuit_number is None else circuit_number
        self.Codelist = []
        self.matrixlist = []
        self.tensorlist = []
        self.qasmlist = [self.Qr.qasmcode, self.Cr.qasmcode]
        self.idgate = np.array([[1., 0.], [0., 1.]])


    def __str__(self):
        return str(self._draw(output='text'))
    def _draw(self,scale=0.7,
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
        return circuit_drawer(qasm,scale=0.7,
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
            assert other.Qr.name not in q_dict, other.Qr.name + " is already used. Please use a different name."
            q_dict.update([(other.Qr.name, self.circuit_number+1)])
            Q = QuantumRegister(q_num)
            Q.numlist = q_numlist
            Q.dict = q_dict
            Q.qasmcode = self.Qr.qasmcode + other.Qr.qasmcode

            c_dict = self.Cr.dict
            C = self.Cr
            if other.Cr.name not in c_dict:
                c_dict.update([(other.Cr.name, self.circuit_number+1)])
                c_num = self.Cr.num + other.Cr.num
                c_numlist = self.Cr.numlist
                c_numlist.extend(other.Cr.numlist)
                C = ClassicalRegister(c_num)
                C.numlist = c_numlist
                C.dict = c_dict
                C.qasmcode = self.Cr.qasmcode + other.Cr.qasmcode

            return QuantumCircuit(Q, C, circuit_number=self.circuit_number+1)
        except AssertionError as e:
            raise

    def barrier(self, *args):
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



    def measure_exec(self):
        p = [i**2 for i in np.abs(self.Qr.qubit)]
        bit_list = list(format(np.random.choice(self.Qr.q_number, p=p),
                               '0' + str(self.Qr.num) + 'b'))
        return bit_list

    def measure(self, *args):
        if args == ():
            c_name = self.Cr.name
            for qc in range(len(self.Qr.numlist)):
                q_name = [k for k, v in self.Qr.dict.items() if v == qc][0]
                val = self.Qr.dict[q_name]
                for i in range(self.Qr.numlist[val]):
                    q_name, calc_num1 = QuantumCircuit.convert_q_number(self, (q_name, i))
                    self.Codelist.append('QC.Cr.cbit[' + str(calc_num1) + '] = list[' + str(calc_num1) + '];\n')
                    self.qasmlist.append('measure ' + q_name + '[' + str(i) + '] -> '\
                    + c_name + '[' + str(calc_num1) + '];')

        elif type(args[0]) is QuantumRegister and type(args[1]) is ClassicalRegister:
            q_name = args[0].name
            c_name = args[1].name
            val = self.Qr.dict[q_name]
            for i in range(self.Qr.numlist[val]):
                q_name, calc_num1 = QuantumCircuit.convert_q_number(self, (q_name, i))
                c_name, calc_num2 = QuantumCircuit.convert_c_number(self, (c_name, i))
                self.Codelist.append('QC.Cr.cbit[' + str(calc_num2) + '] = list[' + str(calc_num1) + '];\n')
                self.qasmlist.append('measure ' + q_name + '[' + str(i) + '] -> '\
                + c_name + '[' + str(i) + '];')

        elif type(args[0]) is tuple and type(args[1]) is tuple:
            q_name, num1 = args[0]
            q_name, calc_num1 = QuantumCircuit.convert_q_number(self, args[0])
            c_name, num2 = args[1]
            c_name, calc_num2 = QuantumCircuit.convert_c_number(self, args[1])
            #QuantumCircuit.barrier(self)
            self.Codelist.append('QC.Cr.cbit[' + str(calc_num2) + '] = list[' + str(calc_num1) + '];\n')
            self.qasmlist.append('measure ' + q_name + '[' + str(num1) + '] -> '\
            + c_name + '[' + str(num2) + '];')


    def c_if(self, cr, num1, gate, q_num, theta=None, phi=None, lam=None):
        q_name, origin_num2 = q_num
        q_name, num2 = QuantumCircuit.convert_q_number(self, q_num)
        p = [i**2 for i in np.abs(self.Qr.qubit)]
        bit_list = list(format(np.random.choice(self.Qr.q_number, p=p),
                               '0' + str(self.Qr.num) + 'b'))
        val = self.Cr.dict[cr.name]
        num = 0
        for i in range(val):
            num += self.Cr.numlist[i]
        cbit = ''
        for i in range(self.Cr.numlist[val]):
            cbit += str(bit_list[num + i])
        gatemark = str(gate)
        # print(gatemark)
        if int(cbit, 2) == num1:
            if (theta is None) and (phi is None) and (lam is None):
                gatemark = gate(q_num)
            elif (phi is None) and (lam is None):
                gatemark = gate(theta, q_num)
            elif lam is None:
                gatemark = gate(theta, phi, q_num)
            else:
                gatemark = gate(theta, phi, lam, q_num)
            del self.qasmlist[-1]
            self.qasmlist.append('if(' + cr.name + '==' + str(num1) + ') '
                                 + gatemark + ' ' + q_name + '[' + str(origin_num2) + '];')

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
        # gate_matrix : 作用させるゲート, num : 目標bit, gatemark : ゲートの記号
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

    def reset(self, q_num):
        # q_num : Qubit名と目標bit
        name, origin_num = q_num
        name, num = QuantumCircuit.convert_q_number(self, q_num)
        resetgate = np.array([[1., 1.], [0., 0.]])
        gate = QuantumCircuit.gate_base1(self, resetgate, num, 'Reset')
        qubit1 = self.Qr.qubit
        self.Qr.qubit = abs(self.Qr.qubit)**2
        self.Qr.qubit = self.Qr.qubit + 0.j
        qubit2 = self.Qr.qubit
        self.Qr.qubit = np.dot(gate, self.Qr.qubit)
        qubit3 = self.Qr.qubit
        self.Qr.qubit = np.sqrt(self.Qr.qubit)
        qubit4 = self.Qr.qubit
        #self.Qr.qubit = self.Qr.qubit / self.Qr.qubit[num]
        self.matrixlist.append('\n---------------- reset(' + name + '[' + str(origin_num) + ']) ----------------\n'
                               + str(np.array(qubit1).reshape(self.Qr.q_states, 1)) +
                               ' ^2 → \n\n' +
                               str(np.array(qubit2).reshape(self.Qr.q_states, 1)) +
                               '\n\n-----------------------\n\n'
                               + str(gate) + '・\n\n' + str(np.array(qubit2).reshape(self.Qr.q_states, 1)) +
                               ' = \n\n' + str(np.array(qubit3).reshape(self.Qr.q_states, 1)
                                               ) + '\n\n-----------------------\n\n'
                               + str(np.array(qubit3).reshape(self.Qr.q_states, 1)) +
                               ' ^-2 → \n\n' + str(np.array(qubit4).reshape(self.Qr.q_states, 1)))
        self.qasmlist.append('reset ' + name + '[' + str(origin_num) + '];')
        return 'reset'

    def id(self, q_num):
        # q_num : Qubit名と目標bit
        name, origin_num = q_num
        name, num = QuantumCircuit.convert_q_number(self, q_num)
        gate = QuantumCircuit.gate_base1(self, self.idgate, num, 'I')
        qubit = self.Qr.qubit
        self.Qr.qubit = np.dot(gate, self.Qr.qubit)
        self.matrixlist.append('\n---------------- id(' + name + '[' + str(origin_num) + ']) ----------------\n'
                               + str(gate) + '・\n\n' + str(np.array(qubit).reshape(self.Qr.q_states, 1)) +
                               ' = \n\n' + str(np.array(self.Qr.qubit).reshape(self.Qr.q_states, 1)))
        self.qasmlist.append('id ' + name + '[' + str(origin_num) + '];')
        return 'id'

    def x(self, q_num):
        # q_num : Qubit名と目標bit
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
        return 'x'

    def z(self, q_num):
        # q_num : Qubit名と目標bit
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
        return 'z'

    def y(self, q_num):
        # q_num : Qubit名と目標bit
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
        return 'y'

    def h(self, q_num):
        # q_num : Qubit名と目標bit
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
        return 'h'

    def s(self, q_num):
        # q_num : Qubit名と目標bit
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
        return 's'

    def sdg(self, q_num):
        # q_num : Qubit名と目標bit
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
        return 'sdg'

    def t(self, q_num):
        # q_num : Qubit名と目標bit
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
        return 't'

    def tdg(self, q_num):
        # q_num : Qubit名と目標bit
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
        return 'tdg'

    def rx(self, theta, q_num):
        # theta : 角度(rad), q_num : Qubit名と目標bit
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
        return 'rx(' + str(theta) + ')'

    def ry(self, theta, q_num):
        # theta : 角度(rad), q_num : Qubit名と目標bit
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
        return 'ry(' + str(theta) + ')'

    def rz(self, theta, q_num):
        # theta : 角度(rad), q_num : Qubit名と目標bit
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
        return 'rz(' + str(theta) + ')'

    def u1(self, lam, q_num):
        # lam : z角度(rad), q_num : Qubit名と目標bit
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
        return 'u1(' + str(lam) + ')'

    def u2(self, phi, lam, q_num):
        # phi : y角度(rad), lam : z角度(rad), q_num : Qubit名と目標bit
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
        return 'u2(' + str(phi) + ', ' + str(lam) + ')'

    def u3(self, theta, phi, lam, q_num):
        # theta : z角度(rad), phi : y角度(rad), lam : z角度(rad), q_num : Qubit名と目標bit
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
        return 'u3(' + str(theta) + ', ' + str(phi) + ', ' + str(lam) + ')'

    def root_u3(self, theta, phi, lam, q_num):
        # theta : z角度(rad), phi : y角度(rad), lam : z角度(rad), q_num : Qubit名と目標bit
        name, origin_num = q_num
        name, num = QuantumCircuit.convert_q_number(self, q_num)
        u3gate = np.array([[np.cos(theta / 2), -np.exp(1.0j * lam) * np.sin(theta / 2)],
                           [np.exp(1.0j * phi) * np.sin(theta / 2), np.exp(1.0j * (lam + phi)) * np.cos(theta / 2)]])
        rootgate = ((1 + 1.0j) / 2) * self.idgate + ((1 - 1.0j) / 2) * u3gate
        gate = QuantumCircuit.gate_base1(self, rootgate, num, 'Root_U3')
        qubit = self.Qr.qubit
        self.Qr.qubit = np.dot(gate, self.Qr.qubit)
        self.matrixlist.append('\n---------------- root_u3(' + str(theta) + ', ' + str(phi) +
                               ', ' + str(lam) + ', ' + name +
                               '[' + str(origin_num) + ']) ----------------\n'
                               + str(gate) + '・\n\n' + str(np.array(qubit).reshape(self.Qr.q_states, 1)) +
                               ' = \n\n' + str(np.array(self.Qr.qubit).reshape(self.Qr.q_states, 1)))
        # self.qasmlist.append('u3(' + str(theta) + ', ' + str(phi) + ', ' + str(lam) + \
        #') ' + name + '[' + str(origin_num) + '];')
        # return 'root_u3(' + str(theta) + ', ' + str(phi) + ', ' + str(lam) + ')'

    def root_u3dg(self, theta, phi, lam, q_num):
        # theta : z角度(rad), phi : y角度(rad), lam : z角度(rad), q_num : Qubit名と目標bit
        name, origin_num = q_num
        name, num = QuantumCircuit.convert_q_number(self, q_num)
        u3gate = np.array([[np.cos(theta / 2), -np.exp(1.0j * lam) * np.sin(theta / 2)],
                           [np.exp(1.0j * phi) * np.sin(theta / 2), np.exp(1.0j * (lam + phi)) * np.cos(theta / 2)]])
        rootgate = ((1 - 1.0j) / 2) * self.idgate + ((1 + 1.0j) / 2) * u3gate
        gate = QuantumCircuit.gate_base1(self, rootgate, num, 'Root_U3^†')
        qubit = self.Qr.qubit
        self.Qr.qubit = np.dot(gate, self.Qr.qubit)
        self.matrixlist.append('\n---------------- root_u3dg(' + str(theta) + ', ' + str(phi) +
                               ', ' + str(lam) + ', ' + name +
                               '[' + str(origin_num) + ']) ----------------\n'
                               + str(gate) + '・\n\n' + str(np.array(qubit).reshape(self.Qr.q_states, 1)) +
                               ' = \n\n' + str(np.array(self.Qr.qubit).reshape(self.Qr.q_states, 1)))
        # self.qasmlist.append('u3(' + str(theta) + ', ' + str(phi) + ', ' + str(lam) + \
        #') ' + name + '[' + str(origin_num) + '];')
        # return 'root_u3dg(' + str(theta) + ', ' + str(phi) + ', ' + str(lam) + ')'

    def gate_base2(self, gate_matrix, ctl, tgt, gatemark):
        # gate_matrix : 作用させるゲート, ctl : 制御bit, tgt : 目標bit, gatemark : ゲートの記号
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
        # num1 : Qubit名と制御bit, num2 : Qubit名と目標bit
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
        #QuantumCircuit.barrier(self, num1, num2)

    def cy(self, q_num1, q_num2):
        # q_num1 : Qubit名と制御bit, q_num2 : Qubit名と目標bit
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
        #QuantumCircuit.barrier(self, num1, num2)

    def cz(self, q_num1, q_num2):
        # q_num1 : Qubit名と制御bit, q_num2 : Qubit名と目標bit
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
        #QuantumCircuit.barrier(self, num1, num2)

    def ch(self, q_num1, q_num2):
        # q_num1 : Qubit名と制御bit, q_num2 : Qubit名と目標bit
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
        #QuantumCircuit.barrier(self, num1, num2)

    def cs(self, q_num1, q_num2):
        # q_num1 : Qubit名と制御bit, q_num2 : Qubit名と目標bit
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
        #QuantumCircuit.barrier(self, num1, num2)

    def cu1(self, lam, q_num1, q_num2):
        # lam : z角度(rad), q_num1 : Qubit名と制御bit, q_num2 : Qubit名と目標bit
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
        #QuantumCircuit.barrier(self, num1, num2)

    def cu3(self, theta, phi, lam, q_num1, q_num2):
        # theta : z角度(rad), phi : y角度(rad), lam : z角度(rad)
        # q_num1 : Qubit名と制御bit, q_num2 : Qubit名と目標bit
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
        #QuantumCircuit.barrier(self, num1, num2)

    def crz(self, lam, q_num1, q_num2):
        # lam : z角度(rad), q_num1 : Qubit名と制御bit, q_num2 : Qubit名と目標bit
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
        #QuantumCircuit.barrier(self, num1, num2)

    def swap(self, q_num1, q_num2):
        # q_num1, q_num2 : Qubit名と交換するbit
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
        #QuantumCircuit.barrier(self, num1, num2)
        tensor = '\n---------------- SWAP ----------------\n' + str(gate)
        self.tensorlist.append(tensor)

    def ccx(self, q_ctl1, q_ctl2, q_tgt):
        # q_ctl1 : Qubit名と制御bit1, q_ctl2 : Qubit名と制御bit2, q_tgt : Qubit名と目標bit
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
        #QuantumCircuit.barrier(self, ctl1, tgt)
        tensor = '\n---------------- ccX ----------------\n' + str(gate)
        self.tensorlist.append(tensor)

    def cswap(self, q_ctl, q_num1, q_num2):
        # q_ctl : Qubit名と制御bit, q_num1,q_num2 : Qubit名と交換するbit
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
        #QuantumCircuit.barrier(self, ctl, num2)
        tensor = '\n---------------- cSWAP ----------------\n' + str(gate)
        self.tensorlist.append(tensor)
