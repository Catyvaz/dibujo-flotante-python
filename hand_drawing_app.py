import cv2
import mediapipe as mp
import numpy as np

class HandDrawingApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing = False
        self.prev_x = None
        self.prev_y = None
        self.im_aux = None
        self.colores ={'blue': (255, 0, 0), 'green': (0, 255, 0), 'red': (0, 0, 255), 'yellow': (0, 255, 255), 'clear': (29,112,246)}
        self.grosorColor = {'blue': 6, 'green': 1, 'red': 1, 'yellow': 1}
        self.grosor = {'chico': 1, 'medio': 6, 'grande': 1}
        self.color_actual = 'blue'
        self.grosor_actual = 'medio'

    @staticmethod
    def is_finger_up(hand_landmarks, finger_tip, finger_mcp):
        return hand_landmarks[finger_tip].y < hand_landmarks[finger_mcp].y

    def process_frame(self, frame, hand_landmarks, finger_states):
        # Detect index finger position
        if finger_states['index'] and not finger_states['middle'] and not finger_states['ring'] and not finger_states['pinky']:
            x = int(hand_landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
            y = int(hand_landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])
            if self.drawing and self.prev_x is not None and self.prev_y is not None:
                cv2.line(self.im_aux, (self.prev_x, self.prev_y), (x, y), self.colores[self.color_actual], self.grosor[self.grosor_actual])
            self.drawing = True
            self.prev_x, self.prev_y = x, y
            cv2.putText(frame, 'Index Finger Up', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            y_offset = 10
            if 10 < x < 60 and y_offset < y < 50 + y_offset:
                print("Color azul seleccionado, " + self.color_actual)
                self.color_actual = "blue"
                self.grosorColor["blue"] = 6
                self.grosorColor["green"] = 2
                self.grosorColor["red"] = 2
                self.grosorColor["yellow"] = 2
            if 70 < x < 120 and y_offset < y < 50 + y_offset:
                print("Color verde seleccionado " + self.color_actual)
                self.color_actual = "green"
                self.grosorColor["blue"] = 2
                self.grosorColor["green"] = 6
                self.grosorColor["red"] = 2
                self.grosorColor["yellow"] = 2
            if 130 < x < 180 and y_offset < y < 50 + y_offset:
                print("Color rojo seleccionado " + self.color_actual)
                self.color_actual = "red"
                self.grosorColor["blue"] = 2
                self.grosorColor["green"] = 2
                self.grosorColor["red"] = 6
                self.grosorColor["yellow"] = 2
            if 190 < x < 240 and y_offset < y < 50 + y_offset:
                print("Color amarillo seleccionado " + self.color_actual)
                self.color_actual = "yellow"
                self.grosorColor["blue"] = 2
                self.grosorColor["green"] = 2
                self.grosorColor["red"] = 2
                self.grosorColor["yellow"] = 6
                
            if 490 < x < 540 and 0 < y < 50:
                self.grosor_actual = "chico" # Grosor del lápiz/marcador virtual
                self.grosor["chico"] = 6
                self.grosor["medio"] = 1
                self.grosor["grande"] = 1
            if 540 < x < 590 and 0 < "medio" < 50:
                self.grosor_actual = 7 # Grosor del lápiz/marcador virtual
                self.grosor["chico"] = 1
                self.grosor["medio"] = 6
                self.grosor["grande"] = 1
            if 590 < x < 640 and 0 < y < 50:
                self.grosor_actual = "grande" # Grosor del lápiz/marcador virtual
                self.grosor["chico"] = 1
                self.grosor["medio"] = 1
                self.grosor["grande"] = 6
        else:
            self.drawing = False
            self.prev_x, self.prev_y = None, None

    def run(self):
        with self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=1) as hands:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.im_aux is None: 
                    self.im_aux = np.zeros(frame.shape, dtype=np.uint8)
                result = hands.process(rgb_frame)
                
                # Cuadrados dibujados en la parte superior izquierda (representan el color a dibujar), 10 píxeles más abajo
                y_offset = 10
                cv2.rectangle(frame,(10,y_offset),(60,50+y_offset), self.colores["blue"], self.grosorColor["blue"])
                cv2.rectangle(frame,(70,y_offset),(120,50+y_offset),self.colores["green"], self.grosorColor["green"])
                cv2.rectangle(frame,(130,y_offset),(180,50+y_offset),self.colores["red"], self.grosorColor["red"])
                cv2.rectangle(frame,(190,y_offset),(240,50+y_offset),self.colores["yellow"], self.grosorColor["yellow"])

                # Rectángulo superior central, que nos ayudará a limpiar la pantalla
                cv2.rectangle(frame,(300,0),(400,50), self.colores["clear"],1)
                cv2.putText(frame,'Limpiar',(320,20),6,0.6,self.colores["clear"],1,cv2.LINE_AA)
                cv2.putText(frame,'pantalla',(320,40),6,0.6,self.colores["clear"],1,cv2.LINE_AA)

                # Cuadrados dibujados en la parte superior derecha (grosor del marcador para dibujar)
                cv2.rectangle(frame,(490,0),(540,50),(0,0,0), self.grosor['chico'])
                cv2.circle(frame,(515,25),3,(0,0,0),-1)
                cv2.rectangle(frame,(540,0),(590,50),(0,0,0),self.grosor['medio'])
                cv2.circle(frame,(565,25),7,(0,0,0),-1)
                cv2.rectangle(frame,(590,0),(640,50),(0,0,0),self.grosor['grande'])
                cv2.circle(frame,(615,25),11,(0,0,0),-1)
                
                
                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        finger_states = {
                            'index': self.is_finger_up(hand_landmarks.landmark, self.mp_hands.HandLandmark.INDEX_FINGER_TIP, self.mp_hands.HandLandmark.INDEX_FINGER_MCP),
                            'middle': self.is_finger_up(hand_landmarks.landmark, self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP, self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP),
                            'ring': self.is_finger_up(hand_landmarks.landmark, self.mp_hands.HandLandmark.RING_FINGER_TIP, self.mp_hands.HandLandmark.RING_FINGER_MCP),
                            'pinky': self.is_finger_up(hand_landmarks.landmark, self.mp_hands.HandLandmark.PINKY_TIP, self.mp_hands.HandLandmark.PINKY_MCP)
                        }
                        self.process_frame(frame, hand_landmarks.landmark, finger_states)
                
                cv2.imshow('Hand Drawing App', frame)
                cv2.imshow('Drawing Canvas', self.im_aux)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = HandDrawingApp()
    app.run()
