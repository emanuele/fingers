import cv2
import mediapipe as mp
import pyttsx3
import time
import threading
from queue import Queue

# =========================
# Text-to-Speech (threaded)
# =========================
tts_queue = Queue()

def tts_worker():
    engine = pyttsx3.init()
    engine.setProperty("rate", 160)
    engine.setProperty("volume", 1.0)

    while True:
        text = tts_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()

threading.Thread(target=tts_worker, daemon=True).start()

# =========================
# MediaPipe Hands
# =========================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

prev_state = None
last_spoken = 0
SPEAK_INTERVAL = 1.0  # seconds

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        fingers_up = []

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            lm = hand_landmarks.landmark

            # ---- Finger detection ----
            if lm[4].x > lm[3].x:
                fingers_up.append("thumb")
            if lm[8].y < lm[6].y:
                fingers_up.append("index")
            if lm[12].y < lm[10].y:
                fingers_up.append("middle")
            if lm[16].y < lm[14].y:
                fingers_up.append("ring")
            if lm[20].y < lm[18].y:
                fingers_up.append("pinky")

            # ---- Draw landmarks & connections ----
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

        state = tuple(fingers_up)
        now = time.time()

        # =========================
        # Speak only on change
        # =========================
        if state != prev_state and (now - last_spoken) > SPEAK_INTERVAL:
            if len(state) == 1:
                text = state[0]
            elif len(state) > 1:
                text = " and ".join(state)
            else:
                text = "no finger"

            print("Detected:", text, flush=True)
            tts_queue.put(text)

            prev_state = state
            last_spoken = now

        # =========================
        # Show video
        # =========================
        cv2.imshow("Hand Segmentation", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
tts_queue.put(None)
