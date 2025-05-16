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
        bg_width = pygame.image.load('./graphics/environment/background.png').get_width()
        self.scale_factor = WINDOW_WIDTH / bg_width

        # sprite setup
        BG(self.all_sprites, self.scale_factor)
        self.plane = Plane(self.all_sprites, self.scale_factor / 2)

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
        self.picam2.preview_configuration.main.size = (320, 320)
        self.picam2.preview_configuration.main.format = "RGB888"
        self.picam2.preview_configuration.align()
        self.picam2.configure("preview")
        self.picam2.start()
        self.latest_nose_position = 0.5
        self.pose_thread_running = True
        self.pose_thread = threading.Thread(target=self.pose_detection_thread)
        self.pose_thread.start()
    

    def pose_detection_thread(self):
        while self.pose_thread_running:
            frame = self.picam2.capture_array()
            results = self.model.predict(frame, imgsz=320, verbose=False)
            try:
                keypoint = results[0].keypoints.xyn[0][0] # 0 is the nose keypoint
                self.latest_nose_position = float(keypoint[1])
            except Exception:
                pass


    def collisions(self):
        if pygame.sprite.spritecollide(self.plane, self.collision_sprites, True, pygame.sprite.collide_mask):
            self.coin_score += 1


    def display_score(self):
        self.time_score = pygame.time.get_ticks() // 1000
        
        time_score_surface = self.font.render(str(self.time_score), True, 'black')
        time_score_rect = time_score_surface.get_rect(topright = (WINDOW_WIDTH - 50, 50))

        coin_score_surface = self.font.render(str(self.coin_score), True, 'black')
        coin_score_rect = coin_score_surface.get_rect(topright = (WINDOW_WIDTH - 50, 100))


        self.display_surface.blit(time_score_surface, time_score_rect)
        self.display_surface.blit(coin_score_surface, coin_score_rect)


    def run(self):
        last_time = time.time()

        while True:
            # delta time
            dt = time.time() - last_time
            last_time = time.time()

            # event handlers
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.pose_thread_running = False
                    self.pose_thread.join()
                    self.picam2.stop()
                    pygame.quit()
                    sys.exit()
                # if event.type == pygame.MOUSEBUTTONDOWN:
                #     self.plane.set_thrust(True)
                # if event.type == pygame.MOUSEBUTTONUP:
                #     self.plane.set_thrust(False)
                if event.type == self.coin_timer:
                    Coin([self.all_sprites, self.collision_sprites], self.scale_factor / 3)

            # plane movement
            if self.latest_nose_position < 0.5:
                # print("Thrusting")
                self.plane.set_thrust(True)
            elif self.latest_nose_position > 0.5:
                # print("Dropping")
                self.plane.set_thrust(False)

            # game logic
            self.display_surface.fill('black')
            self.all_sprites.update(dt)
            self.all_sprites.draw(self.display_surface)
            self.display_score()

            if self.active:
                self.collisions()
            else:
                self.display_score
            
            pygame.display.update()
            self.clock.tick(FRAMERATE)


if __name__ == '__main__':
    game = Game()
    game.run()