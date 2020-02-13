from exercise4/read_xsf_example.py import *
import numpy as np

def periodiser(r0, r1, bbox):
    if not np.shape(r0) == np.shape(r1) == np.shape(bbox):
        print("Bad dimensions!")
        return
    intercepts = []
    len = np.abs(r1-r0)
    for dim in range(np.size(r0)):
        if r1[dim] > bbox[dim]:
            intercept_len = np.abs(len[dim]/bbox[dim])
            intercepts.append((r1-r1)*intercept_len)
    intercepts = sorted(intercepts, key = np.linalg.norm)
    intercepts = intercepts[::-1] # largest to smallest
    if len(intercepts) > 1 :
        return periodiser(intercepts[0], r1, bbox)



