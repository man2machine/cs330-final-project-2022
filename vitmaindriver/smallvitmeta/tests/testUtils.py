#@author:  Faraz, Shahir, Pratyush
#Modified from: https://github.com/aanna0701/SPT_LSA_ViT
import numpy as np
def test_testutils():
    array = np.arange(2048).reshape(4, 512)
    print(array.shape)
    arrayS = array[:,0]
    print(arrayS)