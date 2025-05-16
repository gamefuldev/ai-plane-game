import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import pygame
import numpy as np
import random
from collections import deque
import threading
import time
import math

# Load YOLO model
model = YOLO("yolo11n-pose.pt")

# This sets the resolution of the yolo model being used
imgsz = 96

# Initialize Pygame
pygame.init()

# Game window setup
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 1000
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Nose Space Invaders")

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (179, 0, 255)
BLUE = (164, 224, 0)
YELLOW = (120, 163, 0)

# Game objects
SHIP_WIDTH = 40
SHIP_HEIGHT = 40
ship_x = WINDOW_WIDTH // 2

# Bullet properties
bullets = []
BULLET_SPEED = 20
BULLET_RADIUS = 3

# Enemy properties
enemies = []
ENEMY_WIDTH = 40
ENEMY_HEIGHT = 30
ENEMY_SPEED = 2

# Keypoint detection function
def get_keypoint_position(results, keypoint_num, axis='x'):
    """ 
    Keypoint reference:
        0: nose          5: left_shoulder  10: right_wrist    15: left_ankle
        1: left_eye      6: right_shoulder 11: left_hip       16: right_ankle
        2: right_eye     7: left_elbow     12: right_hip
        3: left_ear     8: right_elbow    13: left_knee
        4: right_ear    9: left_wrist     14: right_knee
    """
    if not 0 <= keypoint_num <= 16:
        raise ValueError("Keypoint number must be between 0 and 16")
    if axis.lower() not in ['x', 'y']:
        raise ValueError("Axis must be 'x' or 'y'")
    
    # Get the keypoint data
    keypoint = results[0].keypoints.xyn[0][keypoint_num]
    
    # Return x or y coordinate based on axis parameter
    return keypoint[0].item() if axis.lower() == 'x' else keypoint[1].item()

def draw_ship(x, y):
    # Draw triangular ship
    points = [
        (x + SHIP_WIDTH//2, y),  # Top point
        (x, y + SHIP_HEIGHT),    # Bottom left
        (x + SHIP_WIDTH, y + SHIP_HEIGHT)  # Bottom right
    ]
    pygame.draw.polygon(screen, GREEN, points)
    
    # Draw engine flames
    flame_points = [
        (x + SHIP_WIDTH//2, y + SHIP_HEIGHT),  # Top
        (x + SHIP_WIDTH//2 - 10, y + SHIP_HEIGHT + 10),  # Left
        (x + SHIP_WIDTH//2 + 10, y + SHIP_HEIGHT + 10)   # Right
    ]
    flame_color = (random.randint(200, 255), random.randint(100, 150), 0)
    pygame.draw.polygon(screen, flame_color, flame_points)

def draw_enemy(x, y):
    # Main body (diamond shape)
    points = [
        (x + ENEMY_WIDTH//2, y),
        (x + ENEMY_WIDTH, y + ENEMY_HEIGHT//2),
        (x + ENEMY_WIDTH//2, y + ENEMY_HEIGHT),
        (x, y + ENEMY_HEIGHT//2)
    ]
    pygame.draw.polygon(screen, RED, points)
    
    # Add angular wings
    wing_points_left = [
        (x, y + ENEMY_HEIGHT//2),
        (x - ENEMY_WIDTH//4, y + ENEMY_HEIGHT//2),
        (x, y + ENEMY_HEIGHT//3)
    ]
    wing_points_right = [
        (x + ENEMY_WIDTH, y + ENEMY_HEIGHT//2),
        (x + ENEMY_WIDTH + ENEMY_WIDTH//4, y + ENEMY_HEIGHT//2),
        (x + ENEMY_WIDTH, y + ENEMY_HEIGHT//3)
    ]
    pygame.draw.polygon(screen, BLUE, wing_points_left)
    pygame.draw.polygon(screen, BLUE, wing_points_right)
    
    # Add menacing "eye"
    eye_color = YELLOW if random.random() > 0.5 else RED
    pygame.draw.polygon(screen, eye_color, [
        (x + ENEMY_WIDTH//2 - 5, y + ENEMY_HEIGHT//2 - 5),
        (x + ENEMY_WIDTH//2 + 5, y + ENEMY_HEIGHT//2 - 5),
        (x + ENEMY_WIDTH//2, y + ENEMY_HEIGHT//2 + 5)
    ])

def draw_explosion(x, y):
    max_radius = 50
    num_shapes = 8
    
    for i in range(num_shapes):
        angle = (2 * math.pi * i) / num_shapes
        points = []
        
        for j in range(3):
            point_angle = angle + (2 * math.pi * j) / 3
            radius = max_radius * (1 - random.random() * 0.3)
            point_x = x + math.cos(point_angle) * radius
            point_y = y + math.sin(point_angle) * radius
            points.append((point_x, point_y))
        
        color = (255, max(100, 255 - i * 20), 0)
        pygame.draw.polygon(screen, color, points)
    
    flash_radius = random.randint(5, 15)
    pygame.draw.circle(screen, WHITE, (int(x), int(y)), flash_radius)

# Position smoothing
SMOOTHING_WINDOW = 2
position_history = deque(maxlen=SMOOTHING_WINDOW)
current_x = WINDOW_WIDTH // 2
for _ in range(SMOOTHING_WINDOW):
    position_history.append(current_x)

# Set up the camera with Picam
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 640)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()


# Shared variables
latest_x_value = 0.5
pose_thread_running = True
explosions = []

def create_enemy():
    return {
        'x': random.randint(0, WINDOW_WIDTH - ENEMY_WIDTH),
        'y': 0
    }

def fire_bullet(x):
    bullets.append({
        'x': x + SHIP_WIDTH//2,
        'y': WINDOW_HEIGHT - SHIP_HEIGHT - 10
    })

def pose_detection_thread():
    global latest_x_value, pose_thread_running
    while pose_thread_running:
        frame = picam2.capture_array()
        results = model.predict(frame, imgsz=imgsz, verbose=False)
        try:
            # Use the new keypoint detection function
            nose_x = get_keypoint_position(results, 0, 'x')  # 0 is the nose keypoint
            latest_x_value = nose_x
        except (IndexError, AttributeError):
            # Keep the previous value if detection fails
            pass
        time.sleep(0.01)

# Start pose detection thread
pose_thread = threading.Thread(target=pose_detection_thread)
pose_thread.start()

# Game loop
running = True
shoot_cooldown = 0
clock = pygame.time.Clock()
score = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False

    # Update ship position with smoothing
    target_x = int(np.interp(latest_x_value, [0.1, 0.9], [WINDOW_WIDTH - SHIP_WIDTH, 0]))  # Inverted range
    position_history.append(target_x)
    current_x = int(sum(position_history) / len(position_history))
    ship_x = np.clip(current_x, 0, WINDOW_WIDTH - SHIP_WIDTH)

    # Game logic
    if random.random() < 0.02:
        enemies.append(create_enemy())

    # Auto-fire bullets
    shoot_cooldown -= 1
    if shoot_cooldown <= 0:
        fire_bullet(ship_x)
        shoot_cooldown = 12

    # Update bullets
    for bullet in bullets[:]:
        bullet['y'] -= BULLET_SPEED
        if bullet['y'] < 0:
            bullets.remove(bullet)

    # Update enemies and check collisions
    for enemy in enemies[:]:
        enemy['y'] += ENEMY_SPEED
        if enemy['y'] > WINDOW_HEIGHT:
            enemies.remove(enemy)
            continue

        # Check collision with bullets
        enemy_center = (enemy['x'] + ENEMY_WIDTH//2, enemy['y'] + ENEMY_HEIGHT//2)
        for bullet in bullets[:]:
            distance = math.hypot(bullet['x'] - enemy_center[0], bullet['y'] - enemy_center[1])
            if distance < ENEMY_WIDTH//2:
                if bullet in bullets: bullets.remove(bullet)
                if enemy in enemies: 
                    enemies.remove(enemy)
                    explosions.append({'x': enemy_center[0], 'y': enemy_center[1], 'timer': 10})
                score += 10
                break

    # Draw everything
    screen.fill((0, 0, 0))
    
    # Draw ship
    draw_ship(ship_x, WINDOW_HEIGHT - SHIP_HEIGHT)
    
    # Draw bullets
    for bullet in bullets:
        pygame.draw.circle(screen, BLUE, (int(bullet['x']), int(bullet['y'])), BULLET_RADIUS + 2)
        pygame.draw.circle(screen, WHITE, (int(bullet['x']), int(bullet['y'])), BULLET_RADIUS)
    
    # Draw enemies
    for enemy in enemies:
        draw_enemy(enemy['x'], enemy['y'])
    
    # Draw explosions
    for explosion in explosions[:]:
        if explosion['timer'] > 0:
            draw_explosion(explosion['x'], explosion['y'])
            explosion['timer'] -= 1
        else:
            explosions.remove(explosion)

    # Draw score
    font = pygame.font.Font(None, 36)
    score_text = font.render(f'SCORE: {score}', True, GREEN)
    pygame.draw.rect(screen, (0, 50, 0), (5, 5, score_text.get_width() + 10, 40))
    screen.blit(score_text, (10, 10))

    pygame.display.flip()
    clock.tick(60)

# Cleanup
pose_thread_running = False
pose_thread.join()
picam2.stop()
pygame.quit()