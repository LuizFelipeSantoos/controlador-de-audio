import cv2
import mediapipe as mp
import pyautogui

# Inicializar o módulo de detecção de mãos do MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Variáveis para controlar o estado de travamento do som e o estado do dedinho
travado = False
dedinho_cima = False

# Função para detectar e rastrear os dedos da mão
def detect_fingers(frame):
    global travado, dedinho_cima

    # Converter o frame para o espaço de cor RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar mãos no frame
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Exibir pontos de referência dos dedos
            for id, landmark in enumerate(hand_landmarks.landmark):
                # Converter as coordenadas normalizadas para pixels
                height, width, _ = frame.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)

                # Desenhar um círculo nos pontos de referência dos dedos
                if id == 4 or id == 8 or id == 12 or id == 16 or id == 20:
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            # Detectar a posição do dedinho em relação ao polegar
            dedinho_y = hand_landmarks.landmark[20].y * height
            polegar_y = hand_landmarks.landmark[4].y * height

            # Verificar o estado de travamento do som
            if travado:
                # Verificar se o dedinho está levantado novamente para destravar o som
                if dedinho_cima and dedinho_y < polegar_y:
                    travado = False
            else:
                # Controlar o volume com base na posição do dedinho
                if dedinho_y > polegar_y:
                    # Levantar o dedinho
                    dedinho_cima = True
                else:
                    # Baixar o dedinho
                    dedinho_cima = False

                    # Aumentar ou diminuir o volume com base na posição do dedo indicador
                    indicador_y = hand_landmarks.landmark[8].y * height
                    if indicador_y > dedinho_y:
                        # Aumentar o volume
                        pyautogui.press('volumeup')
                    else:
                        # Baixar o volume
                        pyautogui.press('volumedown')

                    # Travando o som se o dedinho estiver levantado
                    if dedinho_cima:
                        travado = True

    return frame

# Função principal
def main():
    # Inicializar a captura de vídeo
    cap = cv2.VideoCapture(0)

    while True:
        # Capturar o frame da câmera
        ret, frame = cap.read()

        if ret:
            # Espelhar o frame horizontalmente
            frame = cv2.flip(frame, 1)

            # Detectar e rastrear os dedos da mão
            output_frame = detect_fingers(frame)

            # Exibir a imagem resultante
            cv2.imshow('Hand Detection', output_frame)

        # Encerrar o programa ao pressionar a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar os recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
