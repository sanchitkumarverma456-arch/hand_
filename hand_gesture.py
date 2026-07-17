try:
    import cv2  # type: ignore[import]
except ImportError as e:
    raise ImportError("OpenCV is required. Install it with 'pip install opencv-python'.") from e

try:
    import mediapipe as mp  # type: ignore[import]
except ImportError as e:p
    raise ImportError("MediaPipe is required. Install it with 'pip install mediapipe'.") from e

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)                                                                                                                   
cv2.namedwindow("Hand Gesture Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Hand Gesture Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def detect_gesture(landmarks):
    """Detect hand gestures based on landmarks"""
    if not landmarks:
        return "No Hand"
    
    hand = landmarks
    
    # Get key points
    thumb_tip = hand[4]
    thumb_ip = hand[3]
    index_tip = hand[8]
    index_pip = hand[6]
    middle_tip = hand[12]
    middle_pip = hand[10]
    ring_tip = hand[16]
    ring_pip = hand[14]
    pinky_tip = hand[20]
    pinky_pip = hand[18]
    wrist = hand[0]
    
    # Helper function to check if finger is extended
    def is_finger_extended(tip, pip, wrist):
        return tip[1] < pip[1]  # tip y-coordinate < pip y-coordinate
    
    # Check gestures
    thumb_extended = is_finger_extended(thumb_tip, thumb_ip, wrist)
    index_extended = is_finger_extended(index_tip, index_pip, wrist)
    middle_extended = is_finger_extended(middle_tip, middle_pip, wrist)
    ring_extended = is_finger_extended(ring_tip, ring_pip, wrist)
    pinky_extended = is_finger_extended(pinky_tip, pinky_pip, wrist)
    
    fingers_extended = [index_extended, middle_extended, ring_extended, pinky_extended]
    count_extended = sum(fingers_extended)
    
    # Peace Sign (V)
    if index_extended and middle_extended and not ring_extended and not pinky_extended:
        return "Peace ✌"
    
    # Thumbs Up
    if thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
        if thumb_tip[0] > thumb_ip[0]:
            return "Thumbs Up 👍"
    
    # OK Sign
    if not thumb_extended and not index_extended and middle_extended and ring_extended and pinky_extended:
        return "OK Sign 👌"
    
    # Rock Sign
    if index_extended and pinky_extended and not middle_extended and not ring_extended:
        return "Rock On 🤘"
    
    # Fist
    if not thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
        return "Fist ✊"
    
    # Open Hand / Stop
    if thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended:
        return "Open Hand ✋"
    
    # Point
    if index_extended and not middle_extended and not ring_extended and not pinky_extended:
        return "Pointing 👉"
    
    return f"Fingers: {count_extended}"

print("Hand Gesture Detection")
print("Press 'q' to quit")
print("-" * 40)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    frame_height, frame_width, _ = frame.shape
    
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            
            gesture = detect_gesture(landmarks)
            
            label = handedness.classification[0].label
            cv2.putText(
                frame,
                f"{label}: {gesture}",
                (10, 40 if label == "Right" else 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                2
            )
    
    cv2.putText(frame, "Press 'q' to quit", (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow("Hand Gesture Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
