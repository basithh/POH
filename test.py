from detect import numplatedetect
from recongition import show_results
from segmentation import segment_characters
import cv2
img = cv2.imread("1234.jpeg")
op = numplatedetect(img)
sc = segment_characters(op)
result = show_results(sc)
print(result)