import pygame
from settings import *

class BG(pygame.sprite.Sprite):
    def __init__(self, groups, scale_factor):
        super().__init__(groups)
        bg_image = self.image = pygame.image.load('./graphics/environment/background.png').convert()

        full_height = bg_image.get_height() * scale_factor
        full_width = bg_image.get_width() * scale_factor
        full_sized_image = self.image = pygame.transform.scale(bg_image, (full_width, full_height))

        self.image = pygame.Surface((full_width * 2, full_height))
        self.image.blit(full_sized_image, (0,0))
        self.image.blit(full_sized_image, (full_width, 0))
        self.rect = self.image.get_rect(topleft = (0,0))
        self.pos = pygame.math.Vector2(self.rect.topleft)


    def update(self, dt):
        self.pos.x -= 30 * dt

        if self.rect.centerx <= 0:
            self.pos.x = 0
        
        self.rect.x = round(self.pos.x)

class Plane(pygame.sprite.Sprite):
    def __init__(self, groups, scale_factor):
        super().__init__(groups)

        # image
        self.import_frames(scale_factor)
        self.frame_image = 0
        self.image = self.frames[self.frame_image]

        # rect
        self.rect = self.image.get_rect(midleft = (WINDOW_WIDTH / 20, WINDOW_HEIGHT / 2))
    
    def import_frames(self, scale_factor):
        self.frames = []
        for i in range(3):
            surface = pygame.image.load(f'./graphics/plane/red{i}.png').convert_alpha()
            scaled_surface = pygame.transform.scale(surface, pygame.math.Vector2(surface.get_size()) * scale_factor)
            self.frames.append(scaled_surface)