# import cv2
import numpy as np
# import sys
import matplotlib.pyplot as plt
import base64

from PIL import Image

import io
import sys

a = np.zeros((700, 700))
a[1:5, 1:5] = 1
a[20:30, 20:30] = 2
a[70:80, 70:80] = 3
a[110:120, 110:120] = 4

# 653339
# 1398107
# 5592411
# 22369627

# plt.imshow(a)
# plt.show()

string = np.array2string(a.astype("uint8"))

# bytes = a.astype("uint8")
# s = base64.b64encode(bytes)
# print(sys.getsizeof(str(s)))
# print(len(str(s)))
# exit()
# r = base64.decodebytes(s)
# q = np.frombuffer(r, dtype=np.float64)
# print(s)
# print(r)
# print(q.shape)
# print(sys.getsizeof(s))
# exit()
list_np = a.astype("uint8").tolist()
print(sys.getsizeof(string))
print(sys.getsizeof(list_np))
# print(list_np)
exit()

# base64_image = base64.b64encode(a)
# print(base64_image)
# print(type(base64_image))


# img = base64.decodebytes(base64_image)
# q = np.frombuffer(img, dtype="uint8")
# print(np.unique(q))
