import threading
import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import numpy as np
import pygame, sys, time
from settings import *
from sprites import BG, Plane, Coin

class Game:
    def __init__(self):
        pygame.init()
        self.display_surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption('AI Plane Game')
        self.clock = pygame.time.Clock()
        self.active = True

        # sprite groups
        self.all_sprites = pygame.sprite.Group()
        self.collision_sprites = pygame.sprite.Group()

        # scale factor
        # Load the background image once to get its width for scaling
        bg_image_for_scale = pygame.image.load('./graphics/environment/background.png').convert()
        bg_width = bg_image_for_scale.get_width()
        self.scale_factor = WINDOW_WIDTH / bg_width

        # sprite setup
        # Create BG sprite instance manually, not adding to all_sprites initially
        # The BG class __init__ takes (groups, scale_factor). Pass None for groups.
        self.bg_sprite = BG(None, self.scale_factor) 
        self.plane = Plane(self.all_sprites, self.scale_factor / 2) # Plane uses scale_factor

        # coin timer
        self.coin_timer = pygame.USEREVENT + 1
        pygame.time.set_timer(self.coin_timer, 3000)
    
        # text
        self.font = pygame.font.Font('./graphics/font/Kenney Pixel.ttf', 30)
        self.time_score = 0
        self.coin_score = 0

        # camera setup for YOLO
        self.model = YOLO('yolo11n-pose_ncnn_model')
        self.picam2 = Picamera2()
        self.picam2.preview_configuration.main.size = (320, 320) # Camera resolution
        self.picam2.preview_configuration.main.format = "RGB888"
        self.picam2.preview_configuration.align()
        self.picam2.configure("preview")
        self.picam2.start()
        self.latest_nose_position = 0.5
        self.latest_camera_frame = None # Variable to store the latest frame
        self.pose_thread_running = True
        self.pose_thread = threading.Thread(target=self.pose_detection_thread)
        self.pose_thread.start()
    

    def pose_detection_thread(self):
        while self.pose_thread_running:
            frame = self.picam2.capture_array() # This is a raw RGB frame from Picamera2
            
            results = self.model.predict(frame, imgsz=320, verbose=False)
            
            # Customize plot to exclude boxes and labels
            annotated_frame_bgr = results[0].plot(boxes=False, labels=False) 
            
            annotated_frame_rgb = cv2.cvtColor(annotated_frame_bgr, cv2.COLOR_BGR2RGB)
            self.latest_camera_frame = annotated_frame_rgb.copy()
            
            try:
                keypoint = results[0].keypoints.xyn[0][0] 
                self.latest_nose_position = float(keypoint[1])
            except Exception:
                pass


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

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.pose_thread_running = False
                    if self.pose_thread.is_alive():
                        self.pose_thread.join()
                    self.picam2.stop()
                    pygame.quit()
                    sys.exit()
                if event.type == self.coin_timer:
                    Coin([self.all_sprites, self.collision_sprites], self.scale_factor / 3)


            # plane movement
            if self.latest_nose_position < 0.5:
                self.plane.set_thrust(True)
            elif self.latest_nose_position > 0.5:
                self.plane.set_thrust(False)

            # --- Drawing Start ---
            self.display_surface.fill('black') # Base fill

            # 1. Update and Draw BG sprite (70% opacity)
            self.bg_sprite.update(dt) # Manually update the bg_sprite
            # bg_image_to_draw = self.bg_sprite.image.copy() # Use a copy to set alpha
            # bg_image_to_draw.set_alpha(int(255 * 0.9)) # 70% opacity
            self.display_surface.blit(self.bg_sprite.image, self.bg_sprite.rect)

            # 2. Draw Camera Feed (30% opacity)
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
                scaled_camera_frame.set_alpha(int(255 * 0.15)) # 30% opacity
                
                blit_x = (WINDOW_WIDTH - scaled_width) // 2
                blit_y = (WINDOW_HEIGHT - scaled_height) // 2
                self.display_surface.blit(scaled_camera_frame, (blit_x, blit_y))
            
            # 3. Update and Draw all other game sprites (plane, coins, etc.)
            self.all_sprites.update(dt)
            self.all_sprites.draw(self.display_surface) 
            
            # 4. Display Score
            self.display_score()

            # Collisions
            if self.active:
                self.collisions()
            
            pygame.display.update()
            self.clock.tick(FRAMERATE)


if __name__ == '__main__':
    game = Game()
    game.run()