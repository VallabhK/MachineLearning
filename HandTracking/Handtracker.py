import cv2
import mediapipe as mp

capture = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2, min_detection_confidence=0.4,
                      min_tracking_confidence=0.4)
mpDraw = mp.solutions.drawing_utils


while True:
    success, image = capture.read()
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(image, landmarks, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Image", image)
    cv2.waitKey(1)