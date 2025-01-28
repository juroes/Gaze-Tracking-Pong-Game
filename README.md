# Pong mit Iris- und Blickrichtungserkennung

Dies ist ein einfaches Pong-Spiel, das die Iris- und Blickrichtungserkennung verwendet, um die Bewegung des Paddles zu steuern. Es nutzt OpenCV und Mediapipe zur 
Gesichtserkennung und Augenverfolgung sowie Pygame, um Musik im Hintergrund abzuspielen.

## Funktionen
- Augenerkennung und Blickverfolgung
-   Pong: Ein Game-Klassiker
-   Musik im Hintergrund  : Ein OST vom Spiel Undertale

## Voraussetzungen
- Python 3.x
- OpenCV
- Mediapipe
- Pygame

## Installation
1. Installiere die benötigten Python-Bibliotheken:

    pip install opencv-python mediapipe pygame


2. Lade die Musikdatei   "Undertale OST - Dating Start!.mp3"   und das Startbild   "start_image.jpg"   im selben Verzeichnis wie das Skript.

3. Stelle sicher, dass du eine Webcam angeschlossen hast.

## Verwendung
-   Starten  : Drücke die Leertaste, um das Spiel zu starten.
-   Beenden  : Drücke 'q', um das Spiel zu beenden.
-   Neustart  : Drücke 'r', um das Spiel nach einem Game Over neu zu starten.

## Fehlerbehebung
- Kamera nicht gefunden? Diese Zeile muss angepasst werden:
	webcam = cv2.VideoCapture(Adresse der Webcam)
- Tracking nicht richtig? Für höhere Sensitivität sollte eine niedriger Zahl gewählt werden, so auch umgekehrt. 
  Der <offset> hat sich zwischen 0.12 und 0.16 als ideal erwiesen, es kann aber varrieren aufgrund des Augenabstands und der Abstand von Gesicht und Kamera:
	    left_ratio = left_iris_center[0] / w - <offset>
    	    right_ratio = right_iris_center[0] / w + <offset>
- Das Bild ist spiegelverkehrt? Ob dies der Fall ist, hängt von der Kamera ab. In Zeile 160 soll dieser Codesnippet eingefügt werden:
	frame = cv2.flip(frame, 1) 


Das Spiel endet, wenn der Ball den unteren Rand des Bildschirms erreicht. Der Score wird bei jedem Treffer der Planke erhöht.
