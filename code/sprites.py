import pygame
from settings import *
from random import randint


class BG(pygame.sprite.Sprite):
    def __init__(self, groups, scale_factor):
        if groups is None:
            super().__init__()  # Initialize without adding to any groups
        else:
            super().__init__(groups) # Initialize and add to the specified group(s)
        
        bg_image = pygame.image.load('./graphics/environment/background.png').convert()

        full_height = bg_image.get_height() * scale_factor
        full_width = bg_image.get_width() * scale_factor
        full_sized_image = pygame.transform.scale(bg_image, (full_width, full_height))

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

class Ground(pygame.sprite.Sprite):
    def __init__(self, groups, scale_factor):
        super().__init__(groups) # Initialize and add to the specified group(s)
        
        ground_image = pygame.image.load('./graphics/ground/ground.png').convert_alpha()

        full_height = ground_image.get_height() * scale_factor
        full_width = ground_image.get_width() * scale_factor
        full_sized_image = pygame.transform.scale(ground_image, (full_width, full_height))

        # MODIFIED: Create the surface with per-pixel alpha
        self.image = pygame.Surface((full_width * 2, full_height), pygame.SRCALPHA) 
        self.image.blit(full_sized_image, (0,0))
        self.image.blit(full_sized_image, (full_width, 0))
        
        self.rect = self.image.get_rect(bottomleft = (0, WINDOW_HEIGHT)) 
        self.pos = pygame.math.Vector2(self.rect.topleft) # Use topleft for self.pos consistency


    def update(self, dt):
        self.pos.x -= 120 * dt # Speed of ground scroll

        if self.rect.centerx <= 0: # Reset position for seamless scrolling
            self.pos.x = 0
        
        self.rect.x = round(self.pos.x)
        # self.rect.y = round(self.pos.y) # Ensure y position is also updated from self.pos if it changes

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

        # mask 
        self.mask = pygame.mask.from_surface(self.image)

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

        # Prevent the plane from going above the top of the screen
        if self.pos.y < 50:
            self.pos.y = 50
            self.direction = 0  # Reset direction to prevent further upward movement

        # Prevent the plane from falling below the bottom of the screen
        if self.pos.y + self.rect.height > WINDOW_HEIGHT - 50:
            self.pos.y = WINDOW_HEIGHT - self.rect.height - 50
            self.direction = 0  # Reset direction to prevent further downward movement

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
        self.mask = pygame.mask.from_surface(self.image)
    
    def import_frames(self, scale_factor):
        self.frames = []
        for i in range(3):
            surface = pygame.image.load(f'./graphics/plane/red{i}.png').convert_alpha()
            scaled_surface = pygame.transform.scale(surface, pygame.math.Vector2(surface.get_size()) * scale_factor)
            self.frames.append(scaled_surface)

class Coin(pygame.sprite.Sprite):
    def __init__(self, groups, scale_factor):
        super().__init__(groups)
        surface = pygame.image.load('./graphics/coins/PNG/Coins/coin_32.png').convert_alpha()
        self.image = pygame.transform.scale(surface, pygame.math.Vector2(surface.get_size()) * scale_factor)
        
        coin_x_pos = WINDOW_WIDTH + randint(10, 50)
        coin_y_pos = WINDOW_HEIGHT / 2 + randint(-200, 200) 
        self.rect = self.image.get_rect(center = (coin_x_pos, coin_y_pos))

        self.pos = pygame.math.Vector2(self.rect.topleft)

        self.mask = pygame.mask.from_surface(self.image)

    def update(self, dt):
        self.pos.x -= 200 * dt
        self.rect.x = round(self.pos.x)

        if self.rect.right <= -100:
            self.kill()


class Cloud(pygame.sprite.Sprite):
    def __init__(self, groups, scale_factor):
        super().__init__(groups)
        rand_cloud = randint(1, 8)
        surface = pygame.image.load(f'./graphics/clouds/cloud{rand_cloud}.png').convert_alpha()
        self.image = pygame.transform.scale(surface, pygame.math.Vector2(surface.get_size()) * scale_factor)
        
        coin_x_pos = WINDOW_WIDTH + randint(10, 50)
        coin_y_pos = WINDOW_HEIGHT / 2 + randint(-200, 200) 
        self.rect = self.image.get_rect(center = (coin_x_pos, coin_y_pos))

        self.pos = pygame.math.Vector2(self.rect.topleft)

        self.mask = pygame.mask.from_surface(self.image)

    def update(self, dt):
        self.pos.x -= 100 * dt
        self.rect.x = round(self.pos.x)

        if self.rect.right <= -100:
            self.kill()