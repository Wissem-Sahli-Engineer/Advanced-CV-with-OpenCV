# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=5, min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

while True:
    test, img = cap.read()
    if not test or img is None:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, 
                                  handLms, 
                                  mpHands.HAND_CONNECTIONS
                                  )

    img = cv2.flip(img, 1)
    cv2.imshow('live', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()