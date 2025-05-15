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
        self.pos.x -= 120 * dt

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
        self.rect = self.image.get_rect(midleft=(WINDOW_WIDTH / 20, WINDOW_HEIGHT / 2))
        self.pos = pygame.math.Vector2(self.rect.topleft)

        # movement
        self.gravity = 200
        self.direction = 0
        self.thrust = -200
        self.is_thrusting = False  # New attribute to track mouse press state

        # rotation
        self.current_rotation = 0  # Current rotation of the plane
        self.target_rotation = 0  # Target rotation of the plane
        self.rotation_speed = 3  # Speed at which the rotation interpolates

    def update(self, dt):
        if self.is_thrusting:  # Apply thrust when the mouse is pressed
            self.apply_thrust()
        self.apply_gravity(dt)
        self.animate(dt)
        self.lerp_rotation(dt)  # Smoothly interpolate rotation
        self.rotate()           # Apply the interpolated rotation to the image

    def set_thrust(self, new_thrust_bool):
        self.is_thrusting = new_thrust_bool

    def apply_gravity(self, dt):
        self.direction += self.gravity * dt
        self.pos.y += self.direction * dt
        self.rect.y = round(self.pos.y)

        # Update target rotation based on the direction
        self.target_rotation = -self.direction * 0.06

    def apply_thrust(self):
        self.direction = self.thrust

    def lerp_rotation(self, dt):
        self.current_rotation += (self.target_rotation - self.current_rotation) * self.rotation_speed * dt

    def animate(self, dt):
        self.frame_image += 10 * dt
        if self.frame_image >= len(self.frames):
            self.frame_image = 0
        self.image = self.frames[int(self.frame_image)]
    
    def rotate(self):
        # Apply the interpolated rotation to the image
        rotated_plane = pygame.transform.rotozoom(self.image, self.current_rotation, 1)
        self.image = rotated_plane
    
    def import_frames(self, scale_factor):
        self.frames = []
        for i in range(3):
            surface = pygame.image.load(f'./graphics/plane/red{i}.png').convert_alpha()
            scaled_surface = pygame.transform.scale(surface, pygame.math.Vector2(surface.get_size()) * scale_factor)
            self.frames.append(scaled_surface)