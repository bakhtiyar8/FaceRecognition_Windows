import numpy as np
import cv2
import time
import os
import datetime
import face_recognition
import numpy as np
import cv2
import datetime

#Method resize - making the window site managable
def resize(resized_frame):
    resized_frame = cv2.resize(resized_frame, (1024, 576))
    return resized_frame
#Face recognition handler
def faceRec(frame0, cap, known_face_encodings, known_face_name, face_locations, face_encodings, face_names, process_this_frame, i):
    small_frame = cv2.resize(frame0,(0,0),fx=0.5,fy=0.5)
    rgb_small_frame = small_frame[:,:,::-1]
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        name_list = []
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings,face_encoding,tolerance=0.5)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_name[best_match_index]
                face_names.append(name)
                #now = datetime.now()

    if len(face_names)==0:
        i=0
            
    process_this_frame = not process_this_frame

    for (top,right,bottom,left), name in zip(face_locations, face_names):
        top*=2
        right*=2
        bottom*=2
        left*=2

        cv2.rectangle(frame0,(left,top),(right,bottom),(0,0,255),2)

        cv2.rectangle(frame0,(left,bottom-35),(right,bottom),(0,0,255),cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame0,name,(left+6,bottom-6), font,1.0,(255,255,255),1)
        time = datetime.datetime.today().strftime("D-%d-H-%H %M %S")
        log = f'Detected face: {name}, Camera: {str(cap)} and time: {time}\n'
        file1 = open("./logs/logs.txt", "a")
        file1.write(log)
        file1.close()
        print(log)
        return frame0
#Video Capture RTCP of Cameras
camera1 = 0
cmaera2 = 'rtsp://admin:password@ip/ch0_0.h264'
cap_0 = cv2.VideoCapture(camera1)
cap_1 = cv2.VideoCapture(cmaera2)


#face recognition
known_face_recognition = []
known_face_name = []
imgBakhtiyar = face_recognition.load_image_file('./faces/photo1.jpg')
imgBakhtiyarEncoding = face_recognition.face_encodings(imgBakhtiyar)[0]
imgShux = face_recognition.load_image_file('./faces/photo3.jpg')
imgShuxEncoding = face_recognition.face_encodings(imgShux)[0]
known_face_encodings = [imgBakhtiyarEncoding, imgShuxEncoding]
known_face_name = ['Bakhtiyar', 'Shukhrat']
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
i = 0


while True:
    # Capture frame-by-frame
    ret0, frame0 = cap_0.read()
    ret1, frame1 = cap_1.read()
    if (ret0):
        faceRec(frame0, camera1, known_face_encodings, known_face_name, face_locations, face_encodings, face_names, process_this_frame, i)
        # Display the resulting frame & resize display window
        cv2.imshow('Camera 1', resize(frame0))
    if (ret1):
        faceRec(frame1, cmaera2, known_face_encodings, known_face_name, face_locations, face_encodings, face_names, process_this_frame, i)
        # Display the resulting frame & resize display window
        cv2.imshow('Camera 2', resize(frame1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap_0.release()
cap_1.release()
cv2.destroyAllWindows()