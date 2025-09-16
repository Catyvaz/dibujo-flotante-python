import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
mp_drawing= mp.solutions.drawing_utils

def is_finger_up(hand_landmarks, finger_tip, finger_mcp):
    return hand_landmarks[finger_tip].y < hand_landmarks[finger_mcp].y

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    finger_state = []*5  # Thumb, Index, Middle, Ring, Pinky
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                finger_states = {
                    'index': is_finger_up(hand_landmarks.landmark, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_MCP),
                    'middle': is_finger_up(hand_landmarks.landmark, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP),
                    'ring': is_finger_up(hand_landmarks.landmark, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_MCP),
                    'pinky': is_finger_up(hand_landmarks.landmark, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_MCP)
                }

                if finger_states['index'] and not finger_states['middle'] and not finger_states['ring'] and not finger_states['pinky']:
                    cv2.putText(frame, 'Index Finger Up', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Hand Tracking', frame)
            if cv2.waitKey(1) & 0xFF == 27:
              break   
cap.release()
cv2.destroyAllWindows()