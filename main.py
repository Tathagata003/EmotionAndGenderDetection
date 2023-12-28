# """
# cnn model for FER import FER2013 for emotion
# ResNet for gender recognition import HDF5 file for age and gender
#
# """
# import cv2 as cv
#
#
#
#
#
#
#
#
# # initialize the face detector
# haar_cascade = cv.CascadeClassifier('haar_face.xml')
#
# # create a new cam object
# cap = cv.VideoCapture(0)
# if not cap.isOpened():
#     print("Cannot open camera")
#
#     exit()
#
# while True:
#     ret, frame = cap.read()
#
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)
#
#     for (x, y, w, h) in faces_rect:
#         cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#     cv.imshow("frame", frame)
#     print(f"number of faces detected: {len(faces_rect)}")
#     if cv.waitKey(0):
#         break
#
#
# cap.release()
#
# cv.destroyAllWindows()