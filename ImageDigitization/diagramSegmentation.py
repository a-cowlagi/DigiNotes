# -*- coding: utf-8 -*-
import cv2
from google.colab.patches import cv2_imshow

def get_region_of_interest(img, pad=3, num_diagrams=10):
    # three parameters to findContours:
      # 1.) Image
      # 2.) Contour retrieval mode
      # 3.) Contour approximation method
       
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_area = 1 # originally 0
    dictionary = {}
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        rect_area = w * h
        dictionary[rect_area] = (x, y, w, h)
    keyset = set()
    to_return = []
    for __ in range(num_diagrams):
      max_area = 0
      for key in dictionary:
        if key > max_area and key not in keyset:
          # remove if breaks:
          x, y, w, h = dictionary[key]
          good = True
          for k in dictionary:
            value = dictionary[k]
            if (value[0] > w and value[0] + value[2] < w and value[1] > h and value[1] + value[3] < h):
              good = False
          if good:
            dim = dictionary[key]
            max_area = key
      keyset.add(max_area)
      to_return.append(dim)
    return to_return
img = cv2.imread("testThis.png")
cv2_imshow(img)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
img_canny = cv2.Canny(img_blur, 50, 50)
img_processed = img_canny
for x,y,w,h in get_region_of_interest(img_processed):
  cv2_imshow(img[y:y + h, x:x + w])
