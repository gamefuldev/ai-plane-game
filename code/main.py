import threading
import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import numpy as np
import pygame, sys, time
from enum import Enum, auto

from settings import *
from sprites import BG, Ground, Plane, Coin, Cloud, Pilot, Obstacle # Add Obstacle

class GameState(Enum):
    WAITING_FOR_PLAYER = auto()
    PLAYER_IN_BOX_TIMER_ACTIVE = auto()
    PLAYING = auto()
    GAME_OVER = auto()

class Game:
    def __init__(self):
        pygame.init()
        self.display_surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.FULLSCREEN)
        pygame.display.set_caption('AI Plane Game')
        self.clock = pygame.time.Clock()
        self.active = True 

        # Game State
        self.state = GameState.WAITING_FOR_PLAYER
        self.player_in_box_duration = 0.0
        self.required_in_box_time = 3.0
        self.all_keypoints_in_target_box = False
        self.target_box_norm = {'x_min': 0.2, 'y_min': 0.1, 'x_max': 0.8, 'y_max': 0.9}

        # sprite groups
        self.all_sprites = pygame.sprite.Group()
        self.coin_sprites = pygame.sprite.Group() # For coin collisions
        self.obstacle_sprites = pygame.sprite.Group() # For obstacle collisions

        # scale factor
        bg_image_for_scale = pygame.image.load('./graphics/environment/background.png').convert()
        bg_width = bg_image_for_scale.get_width()
        self.scale_factor = WINDOW_WIDTH / bg_width

        # sprite setup
        self.bg_sprite = BG(None, self.scale_factor) 
        self.plane = Plane(self.all_sprites, self.scale_factor / 2)
        self.pilot_indicator = Pilot(None, self.scale_factor) 

        # timers
        self.coin_timer = pygame.USEREVENT + 1
        self.cloud_timer = pygame.USEREVENT + 2
        self.obstacle_timer = pygame.USEREVENT + 3 # New timer for obstacles
        
        # text
        self.font = pygame.font.Font('./graphics/font/Kenney Pixel.ttf', 30)
        self.status_font = pygame.font.Font('./graphics/font/Kenney Pixel.ttf', 24) 
        self.game_over_font = pygame.font.Font('./graphics/font/Kenney Pixel.ttf', 50) 
        self.time_score = 0 
        self.coin_score = 0

        # Game Timer
        self.game_play_start_ticks = 0 
        self.game_duration_limit = 30.0 
        self.final_total_score = 0

        # Game Over Timer
        self.game_over_start_ticks = 0
        self.game_over_display_duration = 10.0

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
            # If the game is over, skip camera capture and pose detection
            if self.state == GameState.GAME_OVER:
                time.sleep(0.05)  # Sleep briefly to prevent busy-looping and yield CPU
                continue  # Skip the rest of the loop for this iteration

            try:
                frame = self.picam2.capture_array()
                if frame is None: # Basic check
                    time.sleep(0.05)
                    continue

                results = self.model.predict(frame, imgsz=320, verbose=False)
                
                # Conditional plotting based on whether the camera feed is shown
                if self.state == GameState.WAITING_FOR_PLAYER or self.state == GameState.PLAYER_IN_BOX_TIMER_ACTIVE:
                    if results and hasattr(results[0], 'plot'): # Check if results and plot method exist
                        annotated_frame_bgr = results[0].plot(boxes=False, labels=False)
                        annotated_frame_rgb = cv2.cvtColor(annotated_frame_bgr, cv2.COLOR_BGR2RGB)
                        self.latest_camera_frame = annotated_frame_rgb.copy()
                # else: self.latest_camera_frame is not updated if not in these states, which is intended.
                
                current_all_in_box = False
                # Default nose position if no keypoints are found or processed.
                # This ensures self.latest_nose_position always has a valid float.
                nose_y_val = 0.5 

                if results and results[0].keypoints and results[0].keypoints.xyn.nelement() > 0:
                    keypoints_normalized = results[0].keypoints.xyn[0] 
                    if len(keypoints_normalized) > 0: 
                        # Logic for all_keypoints_in_target_box (used in WAITING/TIMER_ACTIVE states)
                        if self.state == GameState.WAITING_FOR_PLAYER or self.state == GameState.PLAYER_IN_BOX_TIMER_ACTIVE:
                            all_inside = True
                            for kx_tensor, ky_tensor in keypoints_normalized:
                                kx, ky = kx_tensor.item(), ky_tensor.item()
                                if not (self.target_box_norm['x_min'] <= kx <= self.target_box_norm['x_max'] and
                                        self.target_box_norm['y_min'] <= ky <= self.target_box_norm['y_max']):
                                    all_inside = False
                                    break
                            current_all_in_box = all_inside
                        
                        # Logic for latest_nose_position (keypoint 0 is typically the nose)
                        # Ensure at least one keypoint (nose) exists to get its Y-coordinate
                        nose_keypoint_y_tensor = keypoints_normalized[0][1] 
                        nose_y_val = float(nose_keypoint_y_tensor.item())

                self.all_keypoints_in_target_box = current_all_in_box
                self.latest_nose_position = nose_y_val
            
            except Exception as e:
                # print(f"Error in pose_detection_thread: {e}") # Uncomment for debugging
                # Set safe defaults in case of an unexpected error during processing
                self.latest_nose_position = 0.5 
                self.all_keypoints_in_target_box = False
                # self.latest_camera_frame could remain or be cleared
                time.sleep(0.1) # Pause briefly after an error


    def check_coin_collisions(self):
        collided_coins = pygame.sprite.spritecollide(self.plane, self.coin_sprites, True, pygame.sprite.collide_mask)
        self.coin_score += len(collided_coins)

    def check_obstacle_collisions(self):
        if pygame.sprite.spritecollide(self.plane, self.obstacle_sprites, False, pygame.sprite.collide_mask):
            return True
        return False


    def display_score(self):
        # self.time_score is updated in the PLAYING state logic
        time_score_surface = self.font.render(f"Time: {self.time_score}", True, 'white')
        time_score_rect = time_score_surface.get_rect(topleft = (50, 50))

        coin_score_surface = self.font.render(f"Coins: {self.coin_score}", True, 'white')
        coin_score_rect = coin_score_surface.get_rect(topleft = (50, 100))

        self.display_surface.blit(time_score_surface, time_score_rect)
        self.display_surface.blit(coin_score_surface, coin_score_rect)


    def reset_game_for_restart(self):
        """Resets variables for a new game session."""
        self.time_score = 0
        self.coin_score = 0
        self.final_total_score = 0
        self.player_in_box_duration = 0.0
        self.active = True 
        
        # Clear existing coins and obstacles
        for sprite in self.coin_sprites:
            sprite.kill() 
        for sprite in self.obstacle_sprites:
            sprite.kill()

        # Reset plane position (optional)
        # self.plane.reset_position() 

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
                if self.state == GameState.PLAYING:
                    if event.type == self.coin_timer:
                        Coin([self.all_sprites, self.coin_sprites], self.scale_factor / 3)
                    if event.type == self.cloud_timer: # Clouds are visual only, not added to collision groups
                        Cloud(self.all_sprites, self.scale_factor / 3)
                    if event.type == self.obstacle_timer:
                        Obstacle([self.all_sprites, self.obstacle_sprites], self.scale_factor)


            # --- State Management ---
            if self.state == GameState.WAITING_FOR_PLAYER:
                if self.all_keypoints_in_target_box:
                    self.state = GameState.PLAYER_IN_BOX_TIMER_ACTIVE
                    self.player_in_box_duration = 0.0 
                self.plane.set_thrust(False) 
                self.pilot_indicator.set_state(False)
            elif self.state == GameState.PLAYER_IN_BOX_TIMER_ACTIVE:
                if self.all_keypoints_in_target_box:
                    self.player_in_box_duration += dt
                    if self.player_in_box_duration >= self.required_in_box_time:
                        self.state = GameState.PLAYING
                        self.game_play_start_ticks = pygame.time.get_ticks() 
                        self.time_score = 0 
                        self.coin_score = 0 
                        pygame.time.set_timer(self.coin_timer, 3000) 
                        pygame.time.set_timer(self.cloud_timer, 7000) 
                        pygame.time.set_timer(self.obstacle_timer, 5000) # Start obstacle timer (every 5 seconds)
                        self.active = True 
                else: 
                    self.state = GameState.WAITING_FOR_PLAYER
                    self.player_in_box_duration = 0.0
                self.plane.set_thrust(False) 
                self.pilot_indicator.set_state(False)
            
            elif self.state == GameState.PLAYING:
                current_elapsed_play_time = (pygame.time.get_ticks() - self.game_play_start_ticks) // 1000
                self.time_score = current_elapsed_play_time 

                # Plane Movement
                is_thrusting_now = self.latest_nose_position < 0.3
                self.plane.set_thrust(is_thrusting_now)
                self.pilot_indicator.set_state(is_thrusting_now)

                # Check for game over conditions
                game_over_triggered = False
                if current_elapsed_play_time >= self.game_duration_limit:
                    game_over_triggered = True
                
                if self.active and self.check_obstacle_collisions(): # Check obstacle collision if game is active
                    game_over_triggered = True

                if game_over_triggered:
                    self.state = GameState.GAME_OVER
                    self.final_total_score = self.time_score + self.coin_score 
                    self.active = False 
                    pygame.time.set_timer(self.coin_timer, 0) 
                    pygame.time.set_timer(self.cloud_timer, 0)
                    pygame.time.set_timer(self.obstacle_timer, 0) # Stop obstacle timer
                    self.game_over_start_ticks = pygame.time.get_ticks() 
                    self.plane.set_thrust(False) 
                    self.pilot_indicator.set_state(False)
                else:
                    # Only check coin collisions if no game-ending condition met
                    if self.active:
                        self.check_coin_collisions()
            
            elif self.state == GameState.GAME_OVER:
                self.plane.set_thrust(False) 
                self.pilot_indicator.set_state(False) 
                game_over_elapsed_time = (pygame.time.get_ticks() - self.game_over_start_ticks) / 1000.0
                if game_over_elapsed_time >= self.game_over_display_duration:
                    self.reset_game_for_restart()
                    self.state = GameState.WAITING_FOR_PLAYER
                    self.pilot_indicator.set_state(False)


            # --- Drawing Start ---
            self.display_surface.fill('black')

            # 1. Update and Draw BG sprite
            if self.state == GameState.PLAYING:
                self.bg_sprite.update(dt)
            
            bg_image_to_draw = self.bg_sprite.image.copy()
            bg_image_to_draw.set_alpha(int(255 * 0.95)) 
            self.display_surface.blit(bg_image_to_draw, self.bg_sprite.rect)

            # 2. Conditionally Draw Camera Feed and related UI (target box, line)
            if self.state == GameState.WAITING_FOR_PLAYER or self.state == GameState.PLAYER_IN_BOX_TIMER_ACTIVE:
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

                    # Draw horizontal dotted line for thrust threshold (only with camera during setup)
                    line_y_threshold = blit_y + (0.3 * scaled_height) 
                    line_color = (255, 255, 0)
                    dash_length, gap_length, line_thickness = 5, 5, 2
                    current_x_line = blit_x
                    while current_x_line < blit_x + scaled_width:
                        pygame.draw.line(self.display_surface, line_color,
                                        (current_x_line, int(line_y_threshold)),
                                        (min(current_x_line + dash_length, blit_x + scaled_width), int(line_y_threshold)),
                                        line_thickness)
                        current_x_line += dash_length + gap_length

                    # Draw Target Box (already conditional on these states)
                    rect_x_on_screen = blit_x + self.target_box_norm['x_min'] * scaled_width
                    rect_y_on_screen = blit_y + self.target_box_norm['y_min'] * scaled_height
                    rect_w_on_screen = (self.target_box_norm['x_max'] - self.target_box_norm['x_min']) * scaled_width
                    rect_h_on_screen = (self.target_box_norm['y_max'] - self.target_box_norm['y_min']) * scaled_height
                    target_rect_pygame = pygame.Rect(rect_x_on_screen, rect_y_on_screen, rect_w_on_screen, rect_h_on_screen)
                    box_color = (0, 255, 0) if self.all_keypoints_in_target_box else (255, 0, 0) 
                    pygame.draw.rect(self.display_surface, box_color, target_rect_pygame, 3)

            # 3. Update and Draw all other game sprites (plane, ground, coins, obstacles, clouds)
            if self.state == GameState.PLAYING and self.active: 
                self.all_sprites.update(dt)
            self.all_sprites.draw(self.display_surface) # Draw sprites from self.all_sprites

            # 3.5 Draw Pilot Indicator (conditionally, on top of other sprites if PLAYING)
            if self.state == GameState.PLAYING:
                self.display_surface.blit(self.pilot_indicator.image, self.pilot_indicator.rect)
            
            # --- Text and UI Messages (drawn last to be on top) ---
            # 4. Display Score (only if playing)
            if self.state == GameState.PLAYING:
                self.display_score()

            # 5. Display State-Specific Messages
            if self.state == GameState.WAITING_FOR_PLAYER:
                msg_surf = self.status_font.render("Align your body within the box", True, (255,255,255))
                msg_rect = msg_surf.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 60))
                self.display_surface.blit(msg_surf, msg_rect)
            elif self.state == GameState.PLAYER_IN_BOX_TIMER_ACTIVE:
                remaining_time = max(0, self.required_in_box_time - self.player_in_box_duration)
                timer_text = f"Starting in: {remaining_time:.1f}s"
                if not self.all_keypoints_in_target_box: 
                    timer_text = "Hold position in the box!"
                msg_surf = self.status_font.render(timer_text, True, (255,255,255))
                msg_rect = msg_surf.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 60))
                self.display_surface.blit(msg_surf, msg_rect)
            elif self.state == GameState.GAME_OVER:
                game_over_text_surf = self.game_over_font.render("GAME OVER", True, (255, 69, 0)) # OrangeRed
                game_over_text_rect = game_over_text_surf.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 50))
                self.display_surface.blit(game_over_text_surf, game_over_text_rect)

                final_score_str = f"Final Score: {self.final_total_score}"
                final_score_surf = self.font.render(final_score_str, True, (255,255,255))
                final_score_rect = final_score_surf.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 20))
                self.display_surface.blit(final_score_surf, final_score_rect)
            
            pygame.display.update()
            self.clock.tick(FRAMERATE)


if __name__ == '__main__':
    game = Game()
    game.run()