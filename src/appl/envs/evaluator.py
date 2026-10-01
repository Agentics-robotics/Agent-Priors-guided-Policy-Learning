"""Object extent under a world-frame unit quaternion."""
import numpy as np

def extent_wxyz(q, half):
    w,x,y,z = np.asarray(q)/np.linalg.norm(q)
    rotation = np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    return np.abs(rotation) @ np.full(3,half)
