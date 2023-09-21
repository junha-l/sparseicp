import numpy as np
import sparseIcp


def sparse_icp(pcd0, pcd1):
    result = sparseIcp.sparse_icp(pcd0, pcd1)
    rot, trs = np.array(result[0]), np.array(result[1])
    return rot, trs
