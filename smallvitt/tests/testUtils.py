import numpy as np
def test_testutils():
    array = np.arange(2048).reshape(4, 512)
    print(array.shape)
    arrayS = array[:,0]
    print(arrayS)