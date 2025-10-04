import joblib
import cv2
from cvzone.HandTrackingModule import HandDetector


def hand2point(hand_roi, detector):
    ret = []
    hand, frame = detector.findHands(hand_roi, draw=True, flipType=True)
    if hand:
        hand1 = hand[0]
        point = hand1["lmList"]
        fingers1 = detector.fingersUp(hand1)
        for p in point:
            for i in p:
                ret.append(i)
        for f in fingers1:
            ret.append(f)
        return ret, frame
    return None, None


def predictsign(points, loaded_model):
    # Reuse
    predictions = loaded_model.predict([points])
    return predictions


# Load model later
loaded_model = joblib.load(r"models\model_asl.pkl")
cap = cv2.VideoCapture(0)
detector = HandDetector(
    staticMode=True, maxHands=1, modelComplexity=1, detectionCon=0.3, minTrackCon=0.5
)
sentence = ""
while True:
    ret, frame = cap.read()
    if not ret:
        break
    hand, frame = detector.findHands(frame, draw=False, flipType=True)
    cv2.imshow("hand Detection", frame)
    if hand:
        hand1 = hand[0]
        bbox = hand1["bbox"]
        center2 = hand1["center"]
        if cv2.waitKey(1) & 0xFF == ord("s"):
            x, y, w, h = bbox
            height, width = frame.shape[:2]
            x1 = max(0, x - 50)
            y1 = max(0, y - 50)
            x2 = min(width, x + w + 50)
            y2 = min(height, y + h + 50)
            hand_roi = frame[y1:y2, x1:x2]
            cv2.imshow("Hand ROI", hand_roi)
            points, img = hand2point(hand_roi, detector)
            if points == None:
                print("sorry your hand is not clear")
            else:
                p = predictsign(points=points, loaded_model=loaded_model)
                print(p)
                sentence += p[0]
                print(sentence)
                img = cv2.putText(
                    img,
                    sentence,
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("pred", img)
    if cv2.waitKey(1) & 0xFF == ord("d"):
        sentence = sentence[:-1]
        print(sentence)
    if cv2.waitKey(1) & 0xFF == ord(" "):
        sentence += " "
        print(sentence)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
