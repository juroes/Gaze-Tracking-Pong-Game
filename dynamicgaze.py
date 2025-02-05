import cv2
import mediapipe as mp
import random
import pygame

pygame.mixer.init()
pygame.mixer.music.load("Undertale OST - Dating Start!.mp3")
pygame.mixer.music.play(-1)

game_over_sound = pygame.mixer.Sound("gameover.mp3")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

webcam = cv2.VideoCapture(0)
ret, frame = webcam.read()
if not ret:
    print("Fehler beim Zugriff auf die Kamera.")
    exit()

height, width, _ = frame.shape

paddle_width = 100
paddle_height = 20
paddle_y = height - 30
paddle_x = (width - paddle_width) // 2

ball_radius = 15
ball_x = width // 2
ball_y = height // 2
ball_dx = random.choice([-5, 5])
ball_dy = -5

game_over = False
game_started = False

start_image = cv2.imread('start_image.jpg')
start_image = cv2.resize(start_image, (width, height))

score = 0

def detect_iris_and_gaze(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_iris_center = calculate_iris_center(face_landmarks, LEFT_IRIS, frame)
            right_iris_center = calculate_iris_center(face_landmarks, RIGHT_IRIS, frame)

            if left_iris_center:
                cv2.circle(frame, left_iris_center, 2, (155, 169, 111), -1)
            if right_iris_center:
                cv2.circle(frame, right_iris_center, 2, (155, 169, 111), -1)

            if left_iris_center and right_iris_center:
                gaze_direction = estimate_gaze_direction(left_iris_center, right_iris_center, face_landmarks, frame)
                control_paddle(gaze_direction)

    return frame

def calculate_iris_center(face_landmarks, iris_indices, frame):
    h, w, _ = frame.shape
    x_coords = [face_landmarks.landmark[i].x * w for i in iris_indices]
    y_coords = [face_landmarks.landmark[i].y * h for i in iris_indices]
    if x_coords and y_coords:
        return int(sum(x_coords) / len(x_coords)), int(sum(y_coords) / len(y_coords))
    return None

def estimate_gaze_direction(left_iris_center, right_iris_center, face_landmarks, frame):
    h, w, _ = frame.shape
    left_face_edge = face_landmarks.landmark[234]
    right_face_edge = face_landmarks.landmark[454]
    face_width = (right_face_edge.x - left_face_edge.x) * w
    
    if face_width == 0:
        return "Zentrum"

    eye_center_x = (left_iris_center[0] + right_iris_center[0]) / 2
    relative_position = (eye_center_x - left_face_edge.x * w) / face_width

    if relative_position < 0.47:
        return "Links"
    elif relative_position > 0.53:
        return "Rechts"
    else:
        return "Zentrum"

def control_paddle(gaze_direction):
    global paddle_x
    if gaze_direction == "Links":
        paddle_x -= 15
    elif gaze_direction == "Rechts":
        paddle_x += 15

    paddle_x = max(0, min(paddle_x, width - paddle_width))

def move_ball():
    global ball_x, ball_y, ball_dx, ball_dy, paddle_x, game_over, score

    ball_x += ball_dx
    ball_y += ball_dy

    if ball_y - ball_radius <= 0 or ball_y + ball_radius >= height:
        ball_dy = -ball_dy

    if ball_x - ball_radius <= 0 or ball_x + ball_radius >= width:
        ball_dx = -ball_dx

    if paddle_y < ball_y + ball_radius <= paddle_y + paddle_height:
        if paddle_x < ball_x < paddle_x + paddle_width:
            ball_dy = -ball_dy
            score += 1
            pygame.mixer.Sound("boing.mp3").play()

        elif paddle_x <= ball_x - ball_radius <= paddle_x + paddle_width // 3:
            ball_dx = -ball_dx
            score += 1
            pygame.mixer.Sound("boing.mp3").play()

        elif paddle_x + 2 * paddle_width // 3 <= ball_x + ball_radius <= paddle_x + paddle_width:
            ball_dx = -ball_dx
            score += 1
            pygame.mixer.Sound("boing.mp3").play()

    if ball_y + ball_radius >= height:
        game_over = True
        return

def restart_game():
    global ball_x, ball_y, ball_dx, ball_dy, paddle_x, game_over, score

    ball_x = width // 2
    ball_y = height // 2
    ball_dx = random.choice([-5, 5])
    ball_dy = -5
    paddle_x = (width - paddle_width) // 2
    game_over = False
    score = 0

    game_over_sound.stop()
    pygame.mixer.music.stop()
    pygame.mixer.music.play(-1)

while True:
    ret, frame = webcam.read()
    frame = cv2.flip(frame,1)
    if not ret:
        break

    if not game_started:
        cv2.imshow('Pong mit Iris- und Blickrichtungserkennung', start_image)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            game_started = True
        continue

    if game_over:
        game_over_frame = cv2.imread('start_image.jpg')
        game_over_frame = cv2.resize(game_over_frame, (width, height))
        game_over_frame[:] = (155, 169, 111)
        pygame.mixer.music.stop()
        game_over_sound.play()
        cv2.putText(game_over_frame, f'Game Over!', (130, height // 2 - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(game_over_frame, f'Dein Score: {score}', (210, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(game_over_frame, f'Druecke R um neu zu beginnen', (50, height // 2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Pong mit Iris- und Blickrichtungserkennung', game_over_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            restart_game()
        elif key == ord('q'):
            break
    else:
        result_frame = detect_iris_and_gaze(frame)
        move_ball()
        cv2.circle(result_frame, (ball_x, ball_y), ball_radius, (155, 169, 111), -1)
        cv2.rectangle(result_frame, (paddle_x, paddle_y), (paddle_x + paddle_width, paddle_y + paddle_height), (155, 169, 111), -1)
        cv2.putText(result_frame, f'Score: {score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (155, 169, 111), 2, cv2.LINE_AA)
        cv2.imshow('Pong mit Iris- und Blickrichtungserkennung', result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
