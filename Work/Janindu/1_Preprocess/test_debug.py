import cv2
import numpy as np
from skimage.morphology import medial_axis
from scipy.ndimage import distance_transform_edt

# Create a dummy binary image
binary = np.zeros((100, 100), dtype=bool)
binary[30:70, 30:70] = True # A square

# Run logic
skeleton, distance = medial_axis(binary, return_distance=True)
mat_debug = np.ones((*binary.shape, 3), dtype=np.uint8) * 255
mat_debug[binary] = [40, 40, 40]

dist, indices = distance_transform_edt(binary, return_indices=True)
skel_y, skel_x = np.where(skeleton)
nearest_y = indices[0, skel_y, skel_x]
nearest_x = indices[1, skel_y, skel_x]

for i in range(0, len(skel_y), 3): 
    cv2.line(mat_debug, (int(skel_x[i]), int(skel_y[i])), (int(nearest_x[i]), int(nearest_y[i])), (0, 255, 255), 1)

mat_debug[skeleton] = [0, 0, 255]
cv2.imwrite("test_debug_output.png", mat_debug)
