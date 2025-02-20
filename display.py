import pygame
from pygame.locals import DOUBLEBUF

class Display2D(object):
    def __init__(self, W, H):
        pygame.init()
        self.screen = pygame.display.set_mode((W, H), DOUBLEBUF)
        self.surface = pygame.Surface(self.screen.get_size()).convert()

    def paint(self, img):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # RGB dönüştürme ve çizim
        pygame.surfarray.blit_array(self.surface, img.swapaxes(0, 1))
        self.screen.blit(self.surface, (0, 0))
        pygame.display.flip()