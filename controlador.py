import cv2
import mediapipe as mp
import pyautogui

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

def detect_fingers(frame):

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar mãos no frame
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, landmark in enumerate(hand_landmarks.landmark):
                height, width, _ = frame.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)

                # Desenhar um círculo nos pontos de referência dos dedos
                if id == 4 or id == 8 or id == 12 or id == 16 or id == 20:
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            # Detectar a posição do dedo indicador em relação ao polegar
            finger_up = True
            if hand_landmarks.landmark[8].y > hand_landmarks.landmark[6].y:
                finger_up = False

            
            if finger_up:
                pyautogui.press('volumeup')
            else:
                pyautogui.press('volumedown')

    return frame


def main():

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        if ret:
            frame = cv2.flip(frame, 1)
            output_frame = detect_fingers(frame)
            cv2.imshow('Hand Detection', output_frame)

        # Encerrar o programa ao pressionar a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
