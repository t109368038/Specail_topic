import mediapipe as mp
import threading
import cv2
class deal_imag(threading.Thread):
    def __init__(self):
        super(deal_imag,self).__init__()
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

    def process(self, image, image1, glweidgt):
        with self.mp_hands.Hands(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,max_num_hands=2) as hands:
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image1 = cv2.cvtColor(cv2.flip(image1, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            image1.flags.writeable = False
            image_v = cv2.vconcat([image, image1])
            results = hands.process(image_v)
            # image_v.flags.writeable = True
            # image_v = cv2.cvtColor(image_v, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image_v, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            cv2.imshow('MediaPipe Hands', image_v)
            cv2.waitKey(1)
