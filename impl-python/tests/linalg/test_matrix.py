"""Comprehensive tests for Matrix class."""

import pytest
import numpy as np
from boom.linalg import Matrix, Vector


class TestMatrixConstruction:
    """Test Matrix construction methods."""
    
    def test_empty_matrix(self):
        """Test creating empty matrix."""
        m = Matrix()
        assert m.nrow() == 0
        assert m.ncol() == 0
        assert m.size() == 0
    
    def test_from_dimensions(self):
        """Test creating matrix from dimensions."""
        m = Matrix((3, 4), fill_value=2.5)
        assert m.nrow() == 3
        assert m.ncol() == 4
        assert m.size() == 12
        for i in range(3):
            for j in range(4):
                assert m(i, j) == 2.5
    
    def test_from_dimensions_alt(self):
        """Test creating matrix from dimensions (alternative syntax)."""
        m = Matrix(2, ncol=3, fill_value=1.5)
        assert m.nrow() == 2
        assert m.ncol() == 3
        for i in range(2):
            for j in range(3):
                assert m(i, j) == 1.5
    
    def test_from_list_of_lists(self):
        """Test creating matrix from list of lists."""
        data = [[1, 2, 3], [4, 5, 6]]
        m = Matrix(data)
        assert m.nrow() == 2
        assert m.ncol() == 3
        assert m(0, 0) == 1.0
        assert m(1, 2) == 6.0
    
    def test_from_numpy(self):
        """Test creating matrix from numpy array."""
        arr = np.array([[1.5, 2.5], [3.5, 4.5]])
        m = Matrix(arr)
        assert m.nrow() == 2
        assert m.ncol() == 2
        assert m(0, 0) == 1.5
        assert m(1, 1) == 4.5
    
    def test_from_string(self):
        """Test creating matrix from string."""
        m = Matrix("1 2 3 | 4 5 6")
        assert m.nrow() == 2
        assert m.ncol() == 3
        assert m(0, 0) == 1.0
        assert m(1, 2) == 6.0
    
    def test_from_string_custom_delim(self):
        """Test creating matrix from string with custom delimiter."""
        m = Matrix("1 2; 3 4", row_delim=";")
        assert m.nrow() == 2
        assert m.ncol() == 2
        assert m(0, 1) == 2.0
        assert m(1, 0) == 3.0
    
    def test_from_vectors_rows(self):
        """Test creating matrix from list of vectors as rows."""
        v1 = Vector([1, 2, 3])
        v2 = Vector([4, 5, 6])
        m = Matrix([v1, v2], byrow=True)
        assert m.nrow() == 2
        assert m.ncol() == 3
        assert m(0, 2) == 3.0
        assert m(1, 0) == 4.0
    
    def test_from_vectors_cols(self):
        """Test creating matrix from list of vectors as columns."""
        v1 = Vector([1, 2])
        v2 = Vector([3, 4])
        v3 = Vector([5, 6])
        m = Matrix([v1, v2, v3], byrow=False)
        assert m.nrow() == 2
        assert m.ncol() == 3
        assert m(0, 0) == 1.0
        assert m(1, 2) == 6.0
    
    def test_copy_constructor(self):
        """Test copy constructor."""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix(m1)
        assert m1 == m2
        assert m1 is not m2
        m2[0, 0] = 999
        assert m1(0, 0) == 1.0  # Original unchanged
    
    def test_reshape_1d_array(self):
        """Test reshaping 1D array into matrix."""
        data = [1, 2, 3, 4, 5, 6]
        m = Matrix(data, ncol=3, byrow=True)
        assert m.nrow() == 2
        assert m.ncol() == 3
        assert m(0, 0) == 1.0
        assert m(1, 2) == 6.0
        
        # Test column-wise reshape
        m2 = Matrix(data, ncol=2, byrow=False)
        assert m2.nrow() == 3
        assert m2.ncol() == 2
        assert m2(0, 0) == 1.0
        assert m2(2, 1) == 6.0


class TestMatrixAccess:
    """Test element access methods."""
    
    def test_call_operator(self):
        """Test call operator for element access."""
        m = Matrix([[10, 20], [30, 40]])
        assert m(0, 0) == 10.0
        assert m(0, 1) == 20.0
        assert m(1, 0) == 30.0
        assert m(1, 1) == 40.0
    
    def test_indexing(self):
        """Test indexing access."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        
        # Single element
        assert m[0, 0] == 1.0
        assert m[1, 2] == 6.0
        
        # Row access
        row = m[0]
        assert isinstance(row, Vector)
        assert list(row) == [1.0, 2.0, 3.0]
        
        # Slice access
        submat = m[0:1, 1:3]
        assert isinstance(submat, Matrix)
        assert submat.nrow() == 1
        assert submat.ncol() == 2
    
    def test_setitem(self):
        """Test setting elements."""
        m = Matrix([[1, 2], [3, 4]])
        m[0, 0] = 99
        assert m(0, 0) == 99.0
        
        # Set row
        m[1] = [7, 8]
        assert m(1, 0) == 7.0
        assert m(1, 1) == 8.0
    
    def test_row_col_access(self):
        """Test row and column access methods."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        
        # Row access
        row1 = m.row(1)
        assert isinstance(row1, Vector)
        assert list(row1) == [4.0, 5.0, 6.0]
        
        # Column access
        col2 = m.col(2)
        assert isinstance(col2, Vector)
        assert list(col2) == [3.0, 6.0]
        
        # Column alias
        col0 = m.column(0)
        assert list(col0) == [1.0, 4.0]
    
    def test_set_row_col(self):
        """Test setting rows and columns."""
        m = Matrix((3, 3), fill_value=0.0)
        
        # Set row
        m.set_row(1, Vector([10, 20, 30]))
        assert m(1, 0) == 10.0
        assert m(1, 2) == 30.0
        
        # Set column
        m.set_col(2, [100, 200, 300])
        assert m(0, 2) == 100.0
        assert m(2, 2) == 300.0
        
        # Set row and column
        m.set_rc(0, 5.0)
        assert m(0, 0) == 5.0
        assert m(0, 1) == 5.0
        assert m(1, 0) == 5.0
    
    def test_diagonal_access(self):
        """Test diagonal access."""
        m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        diag = m.diag()
        assert isinstance(diag, Vector)
        assert list(diag) == [1.0, 5.0, 9.0]
    
    def test_set_diagonal(self):
        """Test setting diagonal."""
        m = Matrix((3, 3), fill_value=1.0)
        
        # Set with vector
        m.set_diag(Vector([10, 20, 30]))
        assert m(0, 0) == 10.0
        assert m(1, 1) == 20.0
        assert m(2, 2) == 30.0
        assert m(0, 1) == 0.0  # Off-diagonal zeroed
        
        # Set with scalar
        m2 = Matrix((2, 2), fill_value=5.0)
        m2.set_diag(99.0, zero_offdiag=False)
        assert m2(0, 0) == 99.0
        assert m2(1, 1) == 99.0
        assert m2(0, 1) == 5.0  # Off-diagonal preserved


class TestMatrixProperties:
    """Test matrix properties and checks."""
    
    def test_size_shape(self):
        """Test size and shape methods."""
        m = Matrix((2, 3), fill_value=1.0)
        assert m.size() == 6
        assert m.nrow() == 2
        assert m.ncol() == 3
        assert m.shape() == (2, 3)
    
    def test_is_square(self):
        """Test square matrix check."""
        m1 = Matrix((3, 3))
        assert m1.is_square()
        
        m2 = Matrix((3, 2))
        assert not m2.is_square()
    
    def test_is_symmetric(self):
        """Test symmetry check."""
        # Symmetric matrix
        m1 = Matrix([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
        assert m1.is_sym()
        
        # Non-symmetric matrix
        m2 = Matrix([[1, 2], [3, 4]])
        assert not m2.is_sym()
        
        # Non-square matrix
        m3 = Matrix([[1, 2, 3], [4, 5, 6]])
        assert not m3.is_sym()
    
    def test_same_dim(self):
        """Test same dimensions check."""
        m1 = Matrix((2, 3))
        m2 = Matrix((2, 3))
        m3 = Matrix((3, 2))
        
        assert m1.same_dim(m2)
        assert not m1.same_dim(m3)
    
    def test_all_finite(self):
        """Test all_finite check."""
        m1 = Matrix([[1, 2], [3, 4]])
        assert m1.all_finite()
        
        m2 = Matrix([[1, np.inf], [3, 4]])
        assert not m2.all_finite()
        
        m3 = Matrix([[1, 2], [np.nan, 4]])
        assert not m3.all_finite()


class TestMatrixArithmetic:
    """Test arithmetic operations."""
    
    def test_add_scalar(self):
        """Test adding scalar to matrix."""
        m = Matrix([[1, 2], [3, 4]])
        m2 = m + 10
        assert m2(0, 0) == 11.0
        assert m2(1, 1) == 14.0
        assert m(0, 0) == 1.0  # Original unchanged
    
    def test_add_matrix(self):
        """Test adding two matrices."""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[10, 20], [30, 40]])
        m3 = m1 + m2
        assert m3(0, 0) == 11.0
        assert m3(1, 1) == 44.0
    
    def test_iadd_operations(self):
        """Test in-place addition."""
        m = Matrix([[1, 2], [3, 4]])
        m += 5
        assert m(0, 0) == 6.0
        assert m(1, 1) == 9.0
        
        m2 = Matrix([[10, 20], [30, 40]])
        m += m2
        assert m(0, 0) == 16.0
        assert m(1, 1) == 49.0
    
    def test_subtract_operations(self):
        """Test subtraction operations."""
        m1 = Matrix([[10, 20], [30, 40]])
        m2 = Matrix([[1, 2], [3, 4]])
        
        m3 = m1 - m2
        assert m3(0, 0) == 9.0
        assert m3(1, 1) == 36.0
        
        m4 = m1 - 5
        assert m4(0, 0) == 5.0
        assert m4(1, 1) == 35.0
    
    def test_multiply_operations(self):
        """Test multiplication operations."""
        m = Matrix([[2, 3], [4, 5]])
        
        # Scalar multiplication
        m2 = m * 3
        assert m2(0, 0) == 6.0
        assert m2(1, 1) == 15.0
        
        # Element-wise multiplication
        m3 = Matrix([[10, 20], [30, 40]])
        m4 = m * m3
        assert m4(0, 0) == 20.0
        assert m4(1, 1) == 200.0
    
    def test_divide_operations(self):
        """Test division operations."""
        m = Matrix([[10, 20], [30, 40]])
        
        # Scalar division
        m2 = m / 5
        assert m2(0, 0) == 2.0
        assert m2(1, 1) == 8.0
        
        # Element-wise division
        m3 = Matrix([[2, 4], [6, 8]])
        m4 = m / m3
        assert m4(0, 0) == 5.0
        assert m4(1, 1) == 5.0
    
    def test_negation(self):
        """Test negation operator."""
        m = Matrix([[1, -2], [-3, 4]])
        m2 = -m
        assert m2(0, 0) == -1.0
        assert m2(0, 1) == 2.0
        assert m2(1, 0) == 3.0
        assert m2(1, 1) == -4.0
    
    def test_right_operators(self):
        """Test right-hand side operators."""
        m = Matrix([[1, 2], [3, 4]])
        
        m2 = 10 + m
        assert m2(0, 0) == 11.0
        assert m2(1, 1) == 14.0
        
        m3 = 2 * m
        assert m3(0, 0) == 2.0
        assert m3(1, 1) == 8.0


class TestMatrixMultiplication:
    """Test matrix multiplication operations."""
    
    def test_matrix_mult(self):
        """Test matrix multiplication."""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])
        
        result = m1 @ m2  # Using @ operator
        assert result(0, 0) == 19.0  # 1*5 + 2*7
        assert result(0, 1) == 22.0  # 1*6 + 2*8
        assert result(1, 0) == 43.0  # 3*5 + 4*7
        assert result(1, 1) == 50.0  # 3*6 + 4*8
        
        # Test mult method
        result2 = m1.mult(m2)
        assert result == result2
    
    def test_matrix_vector_mult(self):
        """Test matrix-vector multiplication."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        v = Vector([10, 20, 30])
        
        result = m @ v
        assert isinstance(result, Vector)
        assert result[0] == 140.0  # 1*10 + 2*20 + 3*30
        assert result[1] == 320.0  # 4*10 + 5*20 + 6*30
        
        # Test mult method
        result2 = m.mult(v)
        assert result == result2
    
    def test_transpose_mult(self):
        """Test transpose multiplication."""
        m1 = Matrix([[1, 2], [3, 4], [5, 6]])  # 3x2
        m2 = Matrix([[7, 8, 9], [10, 11, 12]]) # 2x3
        
        # m1.T @ m2 should be 2x3 (2x3 @ 2x3 won't work)
        # m1.T is 2x3, so we need m2 to be 3xN
        m2_correct = Matrix([[7, 8], [9, 10], [11, 12]])  # 3x2
        result = m1.Tmult(m2_correct)  # 2x3 @ 3x2 = 2x2
        assert result.nrow() == 2
        assert result.ncol() == 2
        assert result(0, 0) == 89.0  # 1*7 + 3*9 + 5*11 = 7 + 27 + 55 = 89
        assert result(0, 1) == 98.0  # 1*8 + 3*10 + 5*12 = 8 + 30 + 60 = 98
        assert result(1, 0) == 116.0  # 2*7 + 4*9 + 6*11 = 14 + 36 + 66 = 116
        assert result(1, 1) == 128.0  # 2*8 + 4*10 + 6*12 = 16 + 40 + 72 = 128
        
        # Test with vector
        v = Vector([7, 8, 9])  # Length 3
        result_v = m1.Tmult(v)   # 2x3 @ 3x1 = 2x1
        assert isinstance(result_v, Vector)
        assert len(result_v) == 2
        assert result_v[0] == 76.0  # 1*7 + 3*8 + 5*9 = 7 + 24 + 45 = 76
        assert result_v[1] == 100.0  # 2*7 + 4*8 + 6*9 = 14 + 32 + 54 = 100
    
    def test_mult_transpose(self):
        """Test multiplication by transpose."""
        m1 = Matrix([[1, 2], [3, 4]])  # 2x2
        m2 = Matrix([[5, 6], [7, 8]])  # 2x2
        
        # m1 @ m2.T should be 2x2
        result = m1.multT(m2)
        assert result.nrow() == 2
        assert result.ncol() == 2
        assert result(0, 0) == 17.0  # [1,2] @ [5,6] = 1*5 + 2*6 = 17
        assert result(0, 1) == 23.0  # [1,2] @ [7,8] = 1*7 + 2*8 = 23
        assert result(1, 0) == 39.0  # [3,4] @ [5,6] = 3*5 + 4*6 = 39
        assert result(1, 1) == 53.0  # [3,4] @ [7,8] = 3*7 + 4*8 = 53
    
    def test_scaled_mult(self):
        """Test scaled multiplication."""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[2, 0], [0, 2]])
        
        result = m1.mult(m2, scal=3.0)
        assert result(0, 0) == 6.0   # 3 * (1*2 + 2*0)
        assert result(1, 1) == 24.0  # 3 * (3*0 + 4*2)


class TestLinearAlgebra:
    """Test linear algebra operations."""
    
    def test_transpose(self):
        """Test transpose operation."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        mt = m.transpose()
        
        assert mt.nrow() == 3
        assert mt.ncol() == 2
        assert mt(0, 0) == 1.0
        assert mt(0, 1) == 4.0
        assert mt(2, 0) == 3.0
        assert mt(2, 1) == 6.0
        
        # Test T property
        mt2 = m.T
        assert mt == mt2
    
    def test_transpose_inplace_square(self):
        """Test in-place transpose for square matrices."""
        m = Matrix([[1, 2], [3, 4]])
        m.transpose_inplace_square()
        
        assert m(0, 0) == 1.0
        assert m(0, 1) == 3.0
        assert m(1, 0) == 2.0
        assert m(1, 1) == 4.0
    
    def test_transpose_inplace_non_square(self):
        """Test in-place transpose fails for non-square."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError):
            m.transpose_inplace_square()
    
    def test_inverse(self):
        """Test matrix inverse."""
        m = Matrix([[2, 1], [1, 1]])  # Invertible matrix
        inv_m = m.inv()
        
        # Check that m @ inv_m ≈ I
        identity = m @ inv_m
        assert abs(identity(0, 0) - 1.0) < 1e-10
        assert abs(identity(1, 1) - 1.0) < 1e-10
        assert abs(identity(0, 1)) < 1e-10
        assert abs(identity(1, 0)) < 1e-10
    
    def test_solve(self):
        """Test solving linear systems."""
        A = Matrix([[2, 1], [1, 3]])
        b = Vector([5, 7])
        
        x = A.solve(b)
        assert isinstance(x, Vector)
        
        # Check solution: A @ x should equal b
        result = A @ x
        np.testing.assert_allclose(result.to_numpy(), b.to_numpy(), rtol=1e-10)
        
        # Test with matrix RHS
        B = Matrix([[5, 1], [7, 2]])
        X = A.solve(B)
        assert isinstance(X, Matrix)
        assert X.nrow() == 2
        assert X.ncol() == 2
    
    def test_determinant(self):
        """Test determinant calculation."""
        m = Matrix([[2, 1], [1, 3]])
        det = m.det()
        assert abs(det - 5.0) < 1e-10  # 2*3 - 1*1 = 5
    
    def test_logdet(self):
        """Test log determinant."""
        m = Matrix([[2, 1], [1, 3]])
        logdet = m.logdet()
        expected = np.log(5.0)
        assert abs(logdet - expected) < 1e-10
    
    def test_trace(self):
        """Test trace calculation."""
        m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        tr = m.trace()
        assert tr == 15.0  # 1 + 5 + 9
    
    def test_inner_product(self):
        """Test inner product X.T @ X."""
        m = Matrix([[1, 2], [3, 4], [5, 6]])  # 3x2
        inner = m.inner()
        
        # Should be 2x2: [[1*1+3*3+5*5, 1*2+3*4+5*6], [2*1+4*3+6*5, 2*2+4*4+6*6]]
        expected = np.array([[35, 44], [44, 56]])
        np.testing.assert_array_equal(inner, expected)
    
    def test_inner_product_weighted(self):
        """Test weighted inner product."""
        m = Matrix([[1, 2], [3, 4]])  # 2x2
        weights = Vector([2, 3])
        
        inner = m.inner(weights)
        # X.T @ diag(weights) @ X
        assert inner.shape == (2, 2)
    
    def test_outer_product(self):
        """Test outer product X @ X.T."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])  # 2x3
        outer = m.outer()
        
        # Should be 2x2
        expected = np.array([[14, 32], [32, 77]])  # [1²+2²+3², 1*4+2*5+3*6], [4*1+5*2+6*3, 4²+5²+6²]
        np.testing.assert_array_equal(outer, expected)
    
    def test_singular_values(self):
        """Test singular values."""
        m = Matrix([[3, 2, 2], [2, 3, -2]])
        sv = m.singular_values()
        
        assert isinstance(sv, Vector)
        assert len(sv) == 2  # min(nrow, ncol)
        assert sv[0] >= sv[1]  # Sorted largest to smallest
    
    def test_condition_number(self):
        """Test condition number."""
        # Well-conditioned matrix
        m1 = Matrix([[2, 0], [0, 2]])
        cond1 = m1.condition_number()
        assert abs(cond1 - 1.0) < 1e-10
        
        # Ill-conditioned matrix
        m2 = Matrix([[1, 1], [1, 1.0001]])
        cond2 = m2.condition_number()
        assert cond2 > 1000  # Should be large
    
    def test_rank(self):
        """Test matrix rank."""
        # Full rank matrix
        m1 = Matrix([[1, 0], [0, 1]])
        assert m1.rank() == 2
        
        # Rank deficient matrix
        m2 = Matrix([[1, 2], [2, 4]])  # Second row = 2 * first row
        assert m2.rank() == 1


class TestMatrixModification:
    """Test matrix modification methods."""
    
    def test_resize(self):
        """Test matrix resizing."""
        m = Matrix([[1, 2], [3, 4]])
        m.resize(3, 3)
        
        assert m.nrow() == 3
        assert m.ncol() == 3
        # Original elements should be preserved where possible
        assert m(0, 0) == 1.0
        assert m(0, 1) == 2.0
    
    def test_rbind(self):
        """Test row binding."""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])
        
        m1.rbind(m2)
        assert m1.nrow() == 4
        assert m1.ncol() == 2
        assert m1(2, 0) == 5.0
        assert m1(3, 1) == 8.0
        
        # Test with vector
        m3 = Matrix([[1, 2]])
        v = Vector([9, 10])
        m3.rbind(v)
        assert m3.nrow() == 2
        assert m3(1, 0) == 9.0
    
    def test_cbind(self):
        """Test column binding."""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])
        
        m1.cbind(m2)
        assert m1.nrow() == 2
        assert m1.ncol() == 4
        assert m1(0, 2) == 5.0
        assert m1(1, 3) == 8.0
        
        # Test with vector
        m3 = Matrix([[1], [2]])
        v = Vector([9, 10])
        m3.cbind(v)
        assert m3.ncol() == 2
        assert m3(0, 1) == 9.0
    
    def test_add_outer(self):
        """Test add outer product."""
        m = Matrix((2, 2), fill_value=0.0)
        x = Vector([1, 2])
        y = Vector([3, 4])
        
        m.add_outer(x, y, w=2.0)
        # Should add 2 * [[1*3, 1*4], [2*3, 2*4]] = [[6, 8], [12, 16]]
        assert m(0, 0) == 6.0
        assert m(0, 1) == 8.0
        assert m(1, 0) == 12.0
        assert m(1, 1) == 16.0


class TestMatrixUtilities:
    """Test utility methods."""
    
    def test_identity(self):
        """Test identity matrix creation."""
        m = Matrix((3, 3))
        identity = m.Id()
        
        assert identity.nrow() == 3
        assert identity.ncol() == 3
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                assert identity(i, j) == expected
    
    def test_identity_non_square(self):
        """Test identity fails for non-square."""
        m = Matrix((2, 3))
        with pytest.raises(ValueError):
            m.Id()
    
    def test_copy(self):
        """Test copy method."""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = m1.copy()
        
        assert m1 == m2
        assert m1 is not m2
        m2[0, 0] = 999
        assert m1(0, 0) == 1.0  # Original unchanged
    
    def test_swap(self):
        """Test swap method."""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])
        
        m1.swap(m2)
        assert m1(0, 0) == 5.0
        assert m1(1, 1) == 8.0
        assert m2(0, 0) == 1.0
        assert m2(1, 1) == 4.0
    
    def test_to_numpy(self):
        """Test conversion to numpy."""
        m = Matrix([[1, 2], [3, 4]])
        arr = m.to_numpy()
        
        assert isinstance(arr, np.ndarray)
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(arr, expected)
        
        # Check it's a copy
        arr[0, 0] = 999
        assert m(0, 0) == 1.0


class TestMatrixRandomization:
    """Test randomization methods."""
    
    def test_randomize(self):
        """Test uniform randomization."""
        m = Matrix((10, 10))
        rng = np.random.RandomState(42)
        m.randomize(rng)
        
        assert m.nrow() == 10
        assert m.ncol() == 10
        
        # All elements should be in [0, 1)
        for i in range(10):
            for j in range(10):
                assert 0 <= m(i, j) < 1
        
        # Should have reasonable mean
        total = sum(m(i, j) for i in range(10) for j in range(10))
        mean = total / 100
        assert 0.4 < mean < 0.6
    
    def test_randomize_gaussian(self):
        """Test Gaussian randomization."""
        m = Matrix((20, 20))
        rng = np.random.RandomState(42)
        m.randomize_gaussian(mean=2.0, sd=0.5, rng=rng)
        
        # Calculate sample mean
        total = sum(m(i, j) for i in range(20) for j in range(20))
        mean = total / 400
        assert 1.8 < mean < 2.2  # Should be close to 2.0


class TestMatrixEquality:
    """Test equality comparisons."""
    
    def test_equal_matrices(self):
        """Test equality of equal matrices."""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[1, 2], [3, 4]])
        assert m1 == m2
    
    def test_unequal_matrices(self):
        """Test inequality of different matrices."""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[1, 2], [3, 5]])
        assert m1 != m2
    
    def test_different_shapes(self):
        """Test matrices of different shapes."""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[1, 2, 3]])
        assert m1 != m2
    
    def test_not_matrix(self):
        """Test comparison with non-matrix."""
        m = Matrix([[1, 2], [3, 4]])
        assert m != [[1, 2], [3, 4]]
        assert m != "not a matrix"


class TestMatrixIteration:
    """Test iteration over matrix."""
    
    def test_iterate_rows(self):
        """Test iterating over rows."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        rows = list(m)
        
        assert len(rows) == 2
        assert isinstance(rows[0], Vector)
        assert isinstance(rows[1], Vector)
        assert list(rows[0]) == [1.0, 2.0, 3.0]
        assert list(rows[1]) == [4.0, 5.0, 6.0]


class TestStringRepresentation:
    """Test string representations."""
    
    def test_str(self):
        """Test string representation."""
        m = Matrix([[1, 2], [3, 4]])
        s = str(m)
        assert "1" in s and "2" in s and "3" in s and "4" in s
    
    def test_repr(self):
        """Test detailed representation."""
        m = Matrix([[1, 2], [3, 4]])
        r = repr(m)
        assert "Matrix" in r
        assert "1" in r and "2" in r and "3" in r and "4" in r