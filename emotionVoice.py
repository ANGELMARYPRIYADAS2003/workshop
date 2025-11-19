import cv2
from deepface import DeepFace
import pyttsx3
import threading

engine = pyttsx3.init()
engine.setProperty('rate', 150)
cap = cv2.VideoCapture(0)

last_emotion = ""
speaking = False  # prevents overlapping speech

def speak(text):
    global speaking
    speaking = True
    engine.say(text)
    engine.runAndWait()
    speaking = False
while True:
    ret, img = cap.read()
    result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
    emotion = result[0]['dominant_emotion']

    cv2.putText(img, f"Emotion: {emotion}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Speak only when emotion changes AND not currently speaking
    if emotion != last_emotion and not speaking:
        threading.Thread(target=speak, args=(f"You look {emotion}",), daemon=True).start()
        last_emotion = emotion
    cv2.imshow("Emotion Recognition", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
