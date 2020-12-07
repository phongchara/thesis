# Classifier demonstration on M6 highway Britain:
import cv2
from imutils.object_detection import non_max_suppression

def perform_classification(video_src, cascade_src):
    cap = cv2.VideoCapture(video_src)
    car_cascade = cv2.CascadeClassifier(cascade_src)
    while True:
        ret, img = cap.read()
        if (type(img) == type(None)):
            print('Video not found')
            break
        image_scaled = cv2.resize(img, None, fx=0.6, fy=0.6)
        gray = cv2.cvtColor(image_scaled, cv2.COLOR_BGR2GRAY)
        cars = car_cascade.detectMultiScale(gray, 1.1, 1) #1.1, 1
        cars = np.array([[x, y, x + w, y + h] for (x, y, w, h) in cars])
        pick = non_max_suppression(cars, probs=None, overlapThresh=0.65)
        for (x, y, w, h) in pick:
            cv2.rectangle(image_scaled, (x, y), (w,  h), (0, 255, 255), 2)
        cv2.imshow('Press ESC key to finish', image_scaled)

            # Nhấn Esc để thoát
        if cv2.waitKey(33) == 27:
            break
    print('Execution finished')
    cv2.destroyAllWindows()
perform_classification('phong_testvideo_02.avi', 'phong_classifier.xml')    # M6 highway Britain