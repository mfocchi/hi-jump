import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def cross_mx(v):
    mx =np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return mx

def invincasadi(A):
    # Determinant of matrix A
    sb1=A[0,0]*((A[1,1]*A[2,2])-(A[1,2]*A[2,1]))
    sb2=A[0,1]*((A[1,0]*A[2,2])-(A[1,2]*A[2,0]))
    sb3=A[0,2]*((A[1,0]*A[2,1])-(A[1,1]*A[2,0]))

    Adetr=sb1-sb2+sb3
#    print(Adetr)
    # Transpose matrix A
    TransA=A.T

    # Find determinant of the minors
    a01=(TransA[1,1]*TransA[2,2])-(TransA[2,1]*TransA[1,2])
    a02=(TransA[1,0]*TransA[2,2])-(TransA[1,2]*TransA[2,0])
    a03=(TransA[1,0]*TransA[2,1])-(TransA[2,0]*TransA[1,1])
    
    a11=(TransA[0,1]*TransA[2,2])-(TransA[0,2]*TransA[2,1])
    a12=(TransA[0,0]*TransA[2,2])-(TransA[0,2]*TransA[2,0])
    a13=(TransA[0,0]*TransA[2,1])-(TransA[0,1]*TransA[2,0])

    a21=(TransA[0,1]*TransA[1,2])-(TransA[1,1]*TransA[0,2])
    a22=(TransA[0,0]*TransA[1,2])-(TransA[0,2]*TransA[1,0])
    a23=(TransA[0,0]*TransA[1,1])-(TransA[0,1]*TransA[1,0])

    # Inverse of determinant
    invAdetr=(float(1)/Adetr)
#    print(invAdetr)
    # Inverse of the matrix A
    invA=np.array([[invAdetr*a01, -invAdetr*a02, invAdetr*a03], [-invAdetr*a11, invAdetr*a12, -invAdetr*a13], [invAdetr*a21, -invAdetr*a22, invAdetr*a23]])

    # Return the matrix
    return invA

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
    else:
        print "Toc: start time not set"
def close_all():
    plt.close('all')