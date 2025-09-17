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
        self.grosor_actual = 7
        self.nombre_grosor = 'medio'
        self.color_changed = False
        self.last_color_change_ts = 0
        self.color_cycle = ['blue','green','red','yellow']
        self.color_index = 0

    def set_stroke_size(self, size_name: str):
        """Actualiza el grosor del trazo y resalta el botón correspondiente.

        size_name: 'chico' | 'medio' | 'grande'
        Mantiene:
          - self.grosor_actual (valor numérico usado por cv2.line)
          - self.nombre_grosor (etiqueta actual)
          - self.grosor (diccionario usado para dibujar el borde de los botones de grosor)
        """
        mapping_valor = {
            'chico': 5,
            'medio': 7,
            'grande': 11
        }
        if size_name not in mapping_valor:
            return
        self.nombre_grosor = size_name
        self.grosor_actual = mapping_valor[size_name]
        # Resalta el seleccionado con grosor 6 en su recuadro y deja 1 en los demás
        self.grosor['chico'] = 6 if size_name == 'chico' else 1
        self.grosor['medio'] = 6 if size_name == 'medio' else 1
        self.grosor['grande'] = 6 if size_name == 'grande' else 1

    def set_color(self, color_name: str, highlight: int = 6, base: int = 2):
        """Cambia el color actual y ajusta SOLO el grosor del borde de los cuadros de color.

        color_name: 'blue' | 'green' | 'red' | 'yellow'
        highlight: grosor del borde para el color seleccionado
        base: grosor del borde para los no seleccionados
        No modifica el grosor del trazo (self.grosor_actual).
        """
        if color_name not in self.colores:
            return
        self.color_actual = color_name
        for c in self.grosorColor.keys():
            self.grosorColor[c] = highlight if c == color_name else base

    @staticmethod
    def is_finger_up(hand_landmarks, finger_tip, finger_mcp):
        if finger_tip == mp.solutions.hands.HandLandmark.THUMB_TIP:
            return hand_landmarks[finger_tip].x < hand_landmarks[finger_mcp].x 
        return hand_landmarks[finger_tip].y < hand_landmarks[finger_mcp].y

    def process_frame(self, frame, hand_landmarks, finger_states):
        # Detect index finger position
        if finger_states['index'] and not finger_states['middle'] and not finger_states['ring'] and not finger_states['pinky']:
            x = int(hand_landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
            y = int(hand_landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])
            if self.drawing and self.prev_x is not None and self.prev_y is not None:
                cv2.line(self.im_aux, (self.prev_x, self.prev_y), (x, y), self.colores[self.color_actual], self.grosor_actual)
            self.drawing = True
            self.prev_x, self.prev_y = x, y
            cv2.putText(frame, 'Index Finger Up', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            y_offset = 10
            if 10 < x < 60 and y_offset < y < 50 + y_offset:
                self.set_color('blue')
            if 70 < x < 120 and y_offset < y < 50 + y_offset:
                self.set_color('green')
            if 130 < x < 180 and y_offset < y < 50 + y_offset:
                self.set_color('red')
            if 190 < x < 240 and y_offset < y < 50 + y_offset:
                self.set_color('yellow')
                
            if 490 < x < 540 and 0 < y < 50:
                self.set_stroke_size('chico')
            if 540 < x < 590 and 0 < y < 50:
                self.set_stroke_size('medio')
            if 590 < x < 640 and 0 < y < 50:
                self.set_stroke_size('grande')    
            if 300 < x < 400 and 0 < y < 50:
                cv2.rectangle(frame,(300,0),(400,50), self.colores["clear"],2)
                cv2.putText(frame,'Limpiar',(320,20),6,0.6,self.colores["clear"],2,cv2.LINE_AA)
                cv2.putText(frame,'pantalla',(320,40),6,0.6,self.colores["clear"],2,cv2.LINE_AA)
                self.im_aux = np.zeros(frame.shape,dtype=np.uint8)
        else:
            self.drawing = False
            self.prev_x, self.prev_y = None, None

    def colour_change(self, finger_states):
        import time
        now = time.time()

        paz = finger_states['index'] and finger_states['middle'] and not finger_states['ring'] and not finger_states['pinky']

        if paz and not self.color_changed:
            # avanzar color
            self.color_index = (self.color_index + 1) % len(self.color_cycle)
            self.color_actual = self.color_cycle[self.color_index]
            print(f"Color cambiado a: {self.color_actual}")
            self.color_changed = True
            self.last_color_change_ts = now
            if self.color_actual == "blue":
                self.set_color('blue')
            elif self.color_actual == "green":
                self.set_color('green')
            elif self.color_actual == "red":
                self.set_color('red')
            elif self.color_actual == "yellow":
                self.set_color('yellow')
        elif not paz:
            # reset para permitir otra activación
            self.color_changed = False
                
    def run(self):
        with self.mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.9, max_num_hands=1) as hands:
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
                            'thumb': self.is_finger_up(hand_landmarks.landmark, self.mp_hands.HandLandmark.THUMB_TIP, self.mp_hands.HandLandmark.THUMB_CMC),
                            'index': self.is_finger_up(hand_landmarks.landmark, self.mp_hands.HandLandmark.INDEX_FINGER_TIP, self.mp_hands.HandLandmark.INDEX_FINGER_MCP),
                            'middle': self.is_finger_up(hand_landmarks.landmark, self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP, self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP),
                            'ring': self.is_finger_up(hand_landmarks.landmark, self.mp_hands.HandLandmark.RING_FINGER_TIP, self.mp_hands.HandLandmark.RING_FINGER_MCP),
                            'pinky': self.is_finger_up(hand_landmarks.landmark, self.mp_hands.HandLandmark.PINKY_TIP, self.mp_hands.HandLandmark.PINKY_MCP)
                        }
                        self.process_frame(frame, hand_landmarks.landmark, finger_states)
                        self.colour_change(finger_states)
                
                imAuxGray = cv2.cvtColor(self.im_aux, cv2.COLOR_BGR2GRAY)
                _, imAuxMask = cv2.threshold(imAuxGray, 20, 255, cv2.THRESH_BINARY)
                thInv = cv2.bitwise_not(imAuxMask)
                frame = cv2.bitwise_and(frame, frame, mask=thInv)
                frame = cv2.add(frame, self.im_aux)
                
                # Redimensionar las ventanas para agrandarlas
                frame_resized = cv2.resize(frame, (1200, 750))  # Duplica el tamaño (de 640x480 a 960x720)
                im_aux_resized = cv2.resize(self.im_aux, (800, 500))               
                
                cv2.imshow('Hand Drawing App', frame_resized)
                cv2.imshow('Drawing Canvas', im_aux_resized)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = HandDrawingApp()
    app.run()
