import threading
import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import numpy as np
import pygame, sys, time
from enum import Enum, auto # Import Enum

from settings import *
from sprites import BG, Plane, Coin

class GameState(Enum):
    WAITING_FOR_PLAYER = auto()
    PLAYER_IN_BOX_TIMER_ACTIVE = auto()
    PLAYING = auto()

class Game:
    def __init__(self):
        pygame.init()
        self.display_surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption('AI Plane Game')
        self.clock = pygame.time.Clock()
        self.active = True # Used for collision detection in PLAYING state

        # Game State
        self.state = GameState.WAITING_FOR_PLAYER
        self.player_in_box_duration = 0.0
        self.required_in_box_time = 3.0
        self.all_keypoints_in_target_box = False
        # Normalized coordinates for the target box (adjust as needed for comfort)
        # (0,0) is top-left of camera feed, (1,1) is bottom-right
        # Make the box taller by adjusting y_min and y_max
        self.target_box_norm = {'x_min': 0.2, 'y_min': 0.1, 'x_max': 0.8, 'y_max': 0.9} # <-- MODIFIED HERE

        # sprite groups
        self.all_sprites = pygame.sprite.Group()
        self.collision_sprites = pygame.sprite.Group()

        # scale factor
        bg_image_for_scale = pygame.image.load('./graphics/environment/background.png').convert()
        bg_width = bg_image_for_scale.get_width()
        self.scale_factor = WINDOW_WIDTH / bg_width

        # sprite setup
        self.bg_sprite = BG(None, self.scale_factor) 
        self.plane = Plane(self.all_sprites, self.scale_factor / 2)

        # coin timer
        self.coin_timer = pygame.USEREVENT + 1
        # Timer will be started when game transitions to PLAYING state

        # text
        self.font = pygame.font.Font('./graphics/font/Kenney Pixel.ttf', 30)
        self.status_font = pygame.font.Font('./graphics/font/Kenney Pixel.ttf', 24) # For status messages
        self.time_score = 0
        self.coin_score = 0

        # camera setup for YOLO
        self.model = YOLO('yolo11n-pose_ncnn_model')
        self.picam2 = Picamera2()
        self.picam2.preview_configuration.main.size = (320, 320)
        self.picam2.preview_configuration.main.format = "RGB888"
        self.picam2.preview_configuration.align()
        self.picam2.configure("preview")
        self.picam2.start()
        self.latest_nose_position = 0.5
        self.latest_camera_frame = None
        self.pose_thread_running = True
        self.pose_thread = threading.Thread(target=self.pose_detection_thread)
        self.pose_thread.start()
    

    def pose_detection_thread(self):
        while self.pose_thread_running:
            frame = self.picam2.capture_array()
            results = self.model.predict(frame, imgsz=320, verbose=False)
            
            annotated_frame_bgr = results[0].plot(boxes=False, labels=False)
            annotated_frame_rgb = cv2.cvtColor(annotated_frame_bgr, cv2.COLOR_BGR2RGB)
            self.latest_camera_frame = annotated_frame_rgb.copy()
            
            current_all_in_box = False
            if results and results[0].keypoints and results[0].keypoints.xyn.nelement() > 0:
                keypoints_normalized = results[0].keypoints.xyn[0] # For the first detected person
                if len(keypoints_normalized) > 0: # Check if there are any keypoints
                    all_inside = True
                    for kx_tensor, ky_tensor in keypoints_normalized:
                        kx, ky = kx_tensor.item(), ky_tensor.item()
                        if not (self.target_box_norm['x_min'] <= kx <= self.target_box_norm['x_max'] and
                                self.target_box_norm['y_min'] <= ky <= self.target_box_norm['y_max']):
                            all_inside = False
                            break
                    current_all_in_box = all_inside
            self.all_keypoints_in_target_box = current_all_in_box
            
            try: # For plane control (nose position)
                if results and results[0].keypoints and results[0].keypoints.xyn.nelement() > 0:
                    nose_keypoint = results[0].keypoints.xyn[0][0] # Index 0 is nose
                    self.latest_nose_position = float(nose_keypoint[1].item())
            except Exception:
                pass # Keep previous nose position or default


    def collisions(self):
        if pygame.sprite.spritecollide(self.plane, self.collision_sprites, True, pygame.sprite.collide_mask):
            self.coin_score += 1


    def display_score(self):
        self.time_score = pygame.time.get_ticks() // 1000
        
        time_score_surface = self.font.render(str(self.time_score), True, 'white') # Changed color to white for visibility
        time_score_rect = time_score_surface.get_rect(topright = (WINDOW_WIDTH - 50, 50))

        coin_score_surface = self.font.render(str(self.coin_score), True, 'white') # Changed color to white
        coin_score_rect = coin_score_surface.get_rect(topright = (WINDOW_WIDTH - 50, 100))

        self.display_surface.blit(time_score_surface, time_score_rect)
        self.display_surface.blit(coin_score_surface, coin_score_rect)


    def run(self):
        last_time = time.time()

        while True:
            dt = time.time() - last_time
            last_time = time.time()

            # --- Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.pose_thread_running = False
                    if self.pose_thread.is_alive():
                        self.pose_thread.join()
                    self.picam2.stop()
                    pygame.quit()
                    sys.exit()
                if self.state == GameState.PLAYING and event.type == self.coin_timer:
                    Coin([self.all_sprites, self.collision_sprites], self.scale_factor / 3)

            # --- State Management ---
            if self.state == GameState.WAITING_FOR_PLAYER:
                if self.all_keypoints_in_target_box:
                    self.state = GameState.PLAYER_IN_BOX_TIMER_ACTIVE
                    self.player_in_box_duration = 0.0 # Reset timer
            elif self.state == GameState.PLAYER_IN_BOX_TIMER_ACTIVE:
                if self.all_keypoints_in_target_box:
                    self.player_in_box_duration += dt
                    if self.player_in_box_duration >= self.required_in_box_time:
                        self.state = GameState.PLAYING
                        pygame.time.set_timer(self.coin_timer, 3000) # Start coin timer
                        self.time_score = 0 
                        self.coin_score = 0 
                else: 
                    self.state = GameState.WAITING_FOR_PLAYER
                    self.player_in_box_duration = 0.0
            
            # --- Plane Movement (only if playing) ---
            if self.state == GameState.PLAYING:
                # MODIFIED THRESHOLD HERE:
                if self.latest_nose_position < 0.3: # Player's nose is above 30% mark -> thrust
                    self.plane.set_thrust(True)
                elif self.latest_nose_position >= 0.3: # Player's nose is at or below 30% mark -> no thrust
                    self.plane.set_thrust(False)
            else: 
                 self.plane.set_thrust(False)


            # --- Drawing Start ---
            self.display_surface.fill('black')

            # 1. Update and Draw BG sprite
            self.bg_sprite.update(dt)
            bg_image_to_draw = self.bg_sprite.image.copy()
            bg_image_to_draw.set_alpha(int(255 * 0.70))
            self.display_surface.blit(bg_image_to_draw, self.bg_sprite.rect)

            # 2. Draw Camera Feed and Target Box/Line
            blit_x, blit_y, scaled_width, scaled_height = 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT 
            if self.latest_camera_frame is not None:
                cam_height, cam_width = self.latest_camera_frame.shape[0], self.latest_camera_frame.shape[1]
                frame_surface = pygame.image.frombuffer(self.latest_camera_frame.tobytes(),
                                                        (cam_width, cam_height), "RGB")
                win_aspect = WINDOW_WIDTH / WINDOW_HEIGHT
                cam_aspect = cam_width / cam_height
                if win_aspect > cam_aspect:
                    scaled_height = WINDOW_HEIGHT
                    scaled_width = int(scaled_height * cam_aspect)
                else:
                    scaled_width = WINDOW_WIDTH
                    scaled_height = int(scaled_width / cam_aspect)
                
                scaled_camera_frame = pygame.transform.scale(frame_surface, (scaled_width, scaled_height))
                scaled_camera_frame.set_alpha(int(255 * 0.30))
                
                blit_x = (WINDOW_WIDTH - scaled_width) // 2
                blit_y = (WINDOW_HEIGHT - scaled_height) // 2
                self.display_surface.blit(scaled_camera_frame, (blit_x, blit_y))

                # Draw horizontal dotted line for thrust threshold (always visible with camera)
                # MODIFIED THRESHOLD HERE:
                line_y_threshold = blit_y + (0.3 * scaled_height) # 0.3 is the normalized threshold
                line_color = (255, 255, 0)
                dash_length, gap_length, line_thickness = 5, 5, 2
                current_x_line = blit_x
                while current_x_line < blit_x + scaled_width:
                    pygame.draw.line(self.display_surface, line_color,
                                     (current_x_line, int(line_y_threshold)),
                                     (min(current_x_line + dash_length, blit_x + scaled_width), int(line_y_threshold)),
                                     line_thickness)
                    current_x_line += dash_length + gap_length

                # Draw Target Box (if not playing)
                if self.state != GameState.PLAYING:
                    rect_x_on_screen = blit_x + self.target_box_norm['x_min'] * scaled_width
                    rect_y_on_screen = blit_y + self.target_box_norm['y_min'] * scaled_height
                    rect_w_on_screen = (self.target_box_norm['x_max'] - self.target_box_norm['x_min']) * scaled_width
                    rect_h_on_screen = (self.target_box_norm['y_max'] - self.target_box_norm['y_min']) * scaled_height
                    target_rect_pygame = pygame.Rect(rect_x_on_screen, rect_y_on_screen, rect_w_on_screen, rect_h_on_screen)
                    box_color = (0, 255, 0) if self.all_keypoints_in_target_box else (255, 0, 0) 
                    pygame.draw.rect(self.display_surface, box_color, target_rect_pygame, 3)

            # --- UI for Waiting/Timer States ---
            if self.state == GameState.WAITING_FOR_PLAYER:
                msg_surf = self.status_font.render("Align your body within the box", True, (255,255,255))
                msg_rect = msg_surf.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 60))
                self.display_surface.blit(msg_surf, msg_rect)
            elif self.state == GameState.PLAYER_IN_BOX_TIMER_ACTIVE:
                remaining_time = max(0, self.required_in_box_time - self.player_in_box_duration)
                timer_text = f"Starting in: {remaining_time:.1f}s"
                if not self.all_keypoints_in_target_box: # Player moved out during countdown
                     timer_text = "Hold position in the box!"
                msg_surf = self.status_font.render(timer_text, True, (255,255,255))
                msg_rect = msg_surf.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 60))
                self.display_surface.blit(msg_surf, msg_rect)
            
            # 3. Update and Draw all other game sprites
            if self.state == GameState.PLAYING:
                self.all_sprites.update(dt)
            # Always draw sprites (plane will be static if not PLAYING and not updated)
            self.all_sprites.draw(self.display_surface)
            
            # 4. Display Score (only if playing)
            if self.state == GameState.PLAYING:
                self.display_score()

            # Collisions (only if playing and active)
            if self.state == GameState.PLAYING and self.active:
                self.collisions()
            
            pygame.display.update()
            self.clock.tick(FRAMERATE)


if __name__ == '__main__':
    game = Game()
    game.run()