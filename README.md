﻿<h1>Hand Gesture Volume Control</h1>

<p>Este é um script em Python que utiliza a biblioteca OpenCV, o MediaPipe e o PyAutoGUI para controlar o volume do sistema operacional com base nos gestos da mão.</p>

<h2>Pré-requisitos</h2>

<p>Certifique-se de ter os seguintes pacotes instalados antes de executar o script:</p>

<ul>
  <li>OpenCV: <code>pip install opencv-python</code></li>
  <li>MediaPipe: <code>pip install mediapipe</code></li>
  <li>PyAutoGUI: <code>pip install pyautogui</code></li>
</ul>

<h2>Como usar</h2>

<ol>
  <li>Execute o script fornecido através do comando: <code>python hand_gesture_volume_control.py</code>.</li>
  <li>A câmera será ativada e exibirá a detecção da mão em tempo real.</li>
  <li>Posicione a mão de forma que os dedos indicador e polegar fiquem estendidos.</li>
  <li>Quando os dedos estiverem estendidos, o volume do sistema será aumentado.</li>
  <li>Quando os dedos estiverem relaxados, o volume do sistema será diminuído.</li>
  <li>Para encerrar o programa, pressione a tecla 'q' no teclado.</li>
</ol>

<h2>Personalização</h2>

<p>Você pode personalizar o script de acordo com suas necessidades:</p>

<ul>
  <li>Para alterar a câmera de entrada, modifique o parâmetro no <code>VideoCapture</code>. Por exemplo, <code>cap = cv2.VideoCapture(1)</code> irá utilizar a segunda câmera conectada.</li>
  <li>Para ajustar a confiança mínima de detecção da mão, altere o valor do parâmetro <code>min_detection_confidence</code> na criação do objeto <code>hands</code>.</li>
</ul>

<h2>Contribuição</h2>

<p>Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e pull requests para melhorias ou correções.</p>

<h2>Licença</h2>

<p>Este projeto está licenciado sob a <a href="LICENSE">MIT License</a>.</p>

<h2>Contato</h2>

<ul>
  <li>Luiz Santos - luizsisantos7@gmail.com</li>
  <li>LinkedIn: <a href="https://www.linkedin.com/in/luiz-felipe-santos-3273881a3/">Luiz Felipe Santos</a></li>
</ul>
