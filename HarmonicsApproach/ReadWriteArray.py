
import numpy as np

def write_array_to_file(array, filename):
    """
    Writes a 2D numpy array to a file. Can be used to store System matrices

    Args:
        array (ndarray): The 2D numpy array to write.
        filename (str): The name of the file to write to.
    """
    with open(filename, 'x') as file:
        for row in array:
            row_string = ' '.join(map(str, row))  # Convert row elements to strings
            file.write(row_string + '\n')  # Write row to file, followed by a newline character

def read_array_from_file(filename):
    """
    Reads a 2D numpy array from a file. Can be used to read stored System matrices

    Args:
        filename (str): The name of the file to read from.

    Returns:
        ndarray: The 2D numpy array read from the file.
    """
    with open(filename, 'r') as file:
        lines = file.readlines()
        array = []
        for line in lines:
            row = list(map(float, line.split()))  # Split line into elements and convert to floats
            array.append(row)
        array = np.array(array)
    return array

def extract_subarray(array, N1, M1, N2, M2):
    """
    Extracts a subarray (J, N1*M1) from a lexicographically ordered array (J, N2*M2). 
    Can be used to turn slice System matrices with a higher harmonics number into System matrices for a lower number of harmonics

    Args:
        array (ndarray): The lexicographically ordered array of shape [J, N2*M2].
        N1 (int): The desired dimension n (n < N).
        M1 (int): The desired dimension m (m < M).
        N2 (int): The original dimension N.
        M2 (int): The original dimension M.

    Returns:
        ndarray: The subarray of shape [J, N1*M1].
        
    """
    J = array.shape[0]
    new_array = np.zeros((J, N1*M1))
    for j in range(J):
        array_j = np.reshape(array[j,:], (N2, M2))
        new_array_j = array_j[:N1, :M1]
        new_array[j, :] = np.reshape(new_array_j, (N1 * M1,))
    return new_array

def TurnSxintoSy(S1, N, M):
    """
    Converts the S matrix for the magnetic field in the x-direction to the S matrix in the y-direction., 
    Can also convert from the y direction to the x direction.
    WARING: Only works when optimizing a field in the Z direction!

    Args:
        S1 (numpy.ndarray): S matrix of shape (J, N * M) in the x-direction.
        N (int): Number of harmonics in the x-direction.
        M (int): Number of harmonics in the y-direction.

    Returns:
        numpy.ndarray: S matrix of shape (J, N * M) in the y-direction.

    """
    J = S1.shape[0]
    S2 = np.zeros((J, N*M))
    for j in range(J):
        S1_j = np.reshape(array[j,:], (N, M))
        S2_j = S1_j.T
        S2[j, :] = np.reshape(new_array_j, (N * M,))
    return S2

if __name__=="__main__":
    x = np.random.normal(size=(10,16))
    write_array_to_file(x, "HarmonicsApproach/testmatrix.txt")
    y = read_array_from_file("HarmonicsApproach/testmatrix.txt")
    print(y)
    print(extract_subarray(y, 2, 2, 4, 4))