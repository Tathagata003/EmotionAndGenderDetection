import cv2
import numpy as np
from keras.models import model_from_json


emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
gender_dict = {0: 'Male', 1: 'Female'}


# load json and create model
json_file = open('emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
# load weights into new model
emotion_model.load_weights("emotion_model.h5")

# gender model from h5
json_file = open('gender_model2.json', 'r')
loaded_gender_model_json = json_file.read()
json_file.close()
gender_model = model_from_json(loaded_gender_model_json)

# load weights into new model
gender_model.load_weights("gender_model2.h5")


print("Loaded model from disk")

# start the webcam feed
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0)

framing = True
while framing:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('haar_face.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        max_emotion_index = int(np.argmax(emotion_prediction))

        gender_pred = gender_model.predict(cropped_img)
        max_gender_index = int(np.argmax(gender_pred[0][0][0]))

        age_pred = gender_model.predict(cropped_img)
        # max_age_index = int(round(age_pred[1][0][0]))
        cv2.putText(frame, f'Emotion: {emotion_dict[max_emotion_index]},Gender:{gender_dict[max_gender_index]}', (x-5, y-20), cv2.FONT_HERSHEY_COMPLEX_SMALL , 1, (255, 0, 0), 2, cv2.LINE_4)

    cv2.imshow('Detection', frame)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()