import cv2
import threading as th
import time
import mediapipe as mp
import sys


class CamCapture(th.Thread):
    def __init__(self, thread_id, name, counter, th_lock, cam_queue=None, save_queue=None, status=0, mode=0,mp4_path=''):
        th.Thread.__init__(self)
        self.threadID = thread_id
        self.name = name
        self.counter = counter
        self.fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
        self.lock = th_lock
        self.mode = mode
        self.cam_queue = cam_queue
        self.save_queue = save_queue
        self.status = status
        self.save_mp4_path = mp4_path

        print('Camera Capture Mode:{}'.format(mode))
        print('========================================')

    def run(self):
        if self.mode == 0:
            # mediapipe
            mp_drawing = mp.solutions.drawing_utils
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(
                min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1)
            self.cam = cv2.VideoCapture(self.counter)
            print("%s: %s\n" % (self.name, time.ctime(time.time())))
            while self.cam.isOpened():
                ret, frame = self.cam.read()
                if not ret:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue
                image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.imshow('MediaPipe Hands' + str(self.counter), image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
            cv2.destroyWindow('MediaPipe Hands' + str(self.counter))
            self.cam.release()
            hands.close()
            print('Close process')
            print("%s: %s" % (self.name, time.ctime(time.time())))
        elif self.mode == 1:
            # no mediapipe
            # cv2.namedWindow(self.name)
            self.cam = cv2.VideoCapture(self.counter)
            sz = (int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))

            self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.vout = cv2.VideoWriter()
            self.vout.open(self.save_mp4_path + 'output'+str(self.counter)+'.mp4', self.fourcc, 20, sz, True)

            self.cam.set(cv2.CAP_PROP_FPS, 20)
            fps = int(self.cam.get(5))
            print('FPS:{}'.format(fps))
            ret, frame = self.cam.read()
            tmp_frame = frame
            tmp_frame = cv2.cvtColor(cv2.flip(tmp_frame, 1), cv2.COLOR_BGR2RGB)

            # self.cam_queue.put(tmp_frame)

            # cv2.imshow(self.name, frame)
            print('Camera is opened')
            print("Camera[%s] open time: %s" % (self.counter, time.ctime(time.time())))
            print('========================================')
            while self.cam.isOpened():
                # print('fps', fps)
                # print(int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)))
                ret, frame = self.cam.read()
                cv2.imshow(self.name, frame)
                tmp_frame = frame
                tmp_frame = cv2.cvtColor(tmp_frame, cv2.COLOR_BGR2RGB)
                if self.status == 1:
                    # print(self.status)
                    self.save_queue.put(tmp_frame)
                    self.cam_queue.put(tmp_frame)
                    self.vout.write(frame)
                # self.cam_queue.put(tmp_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cv2.destroyWindow(self.name)
            self.cam.release()
            self.vout.releas
            print('Close process')
            print("%s: %s" % (self.name, time.ctime(time.time())))
        else:
            raise ValueError('CamCapture does not have this mode.')

    def close(self):
        self.cam.release()
        self.vout.release()