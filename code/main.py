import pygame, sys, time
from settings import *
from sprites import BG, Plane, Coin

class Game:
    def __init__(self):
        pygame.init()
        self.display_surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption('AI Plane Game')
        self.clock = pygame.time.Clock()

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
    

    def run(self):
        last_time = time.time()

        while True:
            # delta time
            dt = time.time() - last_time
            last_time = time.time()

            # event handlers
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.plane.set_thrust(True)
                if event.type == pygame.MOUSEBUTTONUP:
                    self.plane.set_thrust(False)
                if event.type == self.coin_timer:
                    Coin(self.all_sprites, self.scale_factor / 3)

            # game logic
            self.display_surface.fill('black')
            self.all_sprites.update(dt)
            self.all_sprites.draw(self.display_surface)
            pygame.display.update()
            self.clock.tick(FRAMERATE)


if __name__ == '__main__':
    game = Game()
    game.run()