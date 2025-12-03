"""
Final Project
Sampson Mao

Sources:
https://en.wikipedia.org/wiki/Strassen_algorithm
https://stanford.edu/~rezab/classes/cme323/S16/notes/Lecture03/cme323_lec3.pdf
https://cs.stackexchange.com/questions/92666/strassen-algorithm-for-unusal-matrices

Additional sources can be found in the slides
"""


class Matrix:
    def __init__(self, mat):
        self._matrix = mat
        self._nrows = len(mat)
        self._ncols = len(mat[0])

    @property
    def nrows(self):
        return self._nrows

    @property
    def ncols(self):
        return self._ncols

    def __iter__(self):
        for row in self._matrix:
            for entry in row:
                yield entry

    def __str__(self):
        output = ""
        for row_num in range(self.nrows):
            row_str = (
                " ".join(str(val) for val in self._matrix[row_num]) + "\n"
            )
            output += row_str
        return output

    def __add__(self, other):
        if self.nrows != other.nrows or self.ncols != other.ncols:
            raise ValueError(
                "Input matrices are incompatible. Check input dimensions."
            )

        result = [
            [self[i, j] + other[i, j] for j in range(self.ncols)]
            for i in range(self.nrows)
        ]
        c = self.__class__(result)
        return c

    def __sub__(self, other):
        if self.nrows != other.nrows or self.ncols != other.ncols:
            raise ValueError(
                "Input matrices are incompatible. Check input dimensions."
            )

        n = self.nrows
        result = [
            [self[i, j] - other[i, j] for j in range(self.ncols)]
            for i in range(self.nrows)
        ]
        c = self.__class__(result)
        return c

    def __eq__(self, other):
        if self.nrows != other.nrows or self.ncols != other.ncols:
            return False

        for ele1, ele2 in zip(self, other):
            if ele1 != ele2:
                return False
        return True

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._matrix[key]
        elif isinstance(key, tuple) and len(key) == 2:
            row, col = key
            return self._matrix[row][col]
        else:
            raise IndexError("Unsupported indexing method")

    def pad_to_next_pow_two(self, largest_dim):
        if largest_dim & (largest_dim - 1) == 0:
            n = largest_dim
        else:
            n = 2
            while n < largest_dim:
                n *= 2

        expanded = [row + [0] * (n - self.ncols) for row in self._matrix] + [
            [0] * n
        ] * (n - self.nrows)
        return self.__class__(expanded)

    def partition_sq_matrix(self):
        if self.nrows != self.ncols or self.nrows & self.nrows - 1 != 0:
            raise ValueError("Cannot partition with non-power-of-2 shape")

        n = self.nrows
        k = n // 2
        a_11 = self.__class__([self._matrix[row][:k] for row in range(k)])
        a_12 = self.__class__([self._matrix[row][k:] for row in range(k)])
        a_21 = self.__class__([self._matrix[row][:k] for row in range(k, n)])
        a_22 = self.__class__([self._matrix[row][k:] for row in range(k, n)])
        return a_11, a_12, a_21, a_22

    def _recursive_matmul(self, a, b):
        n = a.nrows

        if n == 1:
            return self.__class__([[a[0, 0] * b[0, 0]]])

        a_11, a_12, a_21, a_22 = a.partition_sq_matrix()
        b_11, b_12, b_21, b_22 = b.partition_sq_matrix()

        a_11_b_11 = self._recursive_matmul(a_11, b_11)
        a_12_b_21 = self._recursive_matmul(a_12, b_21)
        a_11_b_12 = self._recursive_matmul(a_11, b_12)
        a_12_b_22 = self._recursive_matmul(a_12, b_22)
        a_21_b_11 = self._recursive_matmul(a_21, b_11)
        a_22_b_21 = self._recursive_matmul(a_22, b_21)
        a_21_b_12 = self._recursive_matmul(a_21, b_12)
        a_22_b_22 = self._recursive_matmul(a_22, b_22)

        c_11 = a_11_b_11 + a_12_b_21
        c_12 = a_11_b_12 + a_12_b_22
        c_21 = a_21_b_11 + a_22_b_21
        c_22 = a_21_b_12 + a_22_b_22

        c = []
        for i in range(n // 2):
            c.append(c_11[i] + c_12[i])
        for i in range(n // 2):
            c.append(c_21[i] + c_22[i])
        return self.__class__(c)

    def __matmul__(self, other):
        if self.ncols != other.nrows:
            raise ValueError(
                "Input matrices are incompatible. Check input dimensions."
            )

        largest_dim = max(self.nrows, self.ncols, other.nrows, other.ncols)

        a = self.pad_to_next_pow_two(largest_dim)
        b = other.pad_to_next_pow_two(largest_dim)

        result = self._recursive_matmul(a, b)

        c = [row[: other.ncols] for row in result._matrix[: self.nrows]]

        return self.__class__(c)


class StrassenMatrix(Matrix):
    def __init__(self, mat):
        super().__init__(mat)

    def _recursive_matmul(self, a, b):
        n = a.nrows

        if n == 1:
            return self.__class__([[a[0, 0] * b[0, 0]]])

        a_11, a_12, a_21, a_22 = a.partition_sq_matrix()
        b_11, b_12, b_21, b_22 = b.partition_sq_matrix()

        m_1 = self._recursive_matmul(a_11 + a_22, b_11 + b_22)
        m_2 = self._recursive_matmul(a_21 + a_22, b_11)
        m_3 = self._recursive_matmul(a_11, b_12 - b_22)
        m_4 = self._recursive_matmul(a_22, b_21 - b_11)
        m_5 = self._recursive_matmul(a_11 + a_12, b_22)
        m_6 = self._recursive_matmul(a_21 - a_11, b_11 + b_12)
        m_7 = self._recursive_matmul(a_12 - a_22, b_21 + b_22)

        c_11 = m_1 + m_4 - m_5 + m_7
        c_12 = m_3 + m_5
        c_21 = m_2 + m_4
        c_22 = m_1 - m_2 + m_3 + m_6

        c = []
        for i in range(n // 2):
            c.append(c_11[i] + c_12[i])
        for i in range(n // 2):
            c.append(c_21[i] + c_22[i])
        return self.__class__(c)
