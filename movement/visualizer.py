import pygame
import numpy as np

movement_file = open("data/joints_s03_e02.txt")
movement_lines = movement_file.readlines()

movement_frames = [[float(v) for v in line.strip().split()[1:]] for line in movement_lines]
movement_frames = np.array(movement_frames)
movement_frames = movement_frames.reshape((-1, 10, 3))

pygame.init()

size = (700, 500)
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Python Visualizer")

_run = True
clock = pygame.time.Clock()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)

min_val = np.amin(movement_frames)
max_val = np.amax(movement_frames)
movement_frames -= min_val
movement_frames *= size[1] / max_val

movement_index = 0
while _run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            carryOn = False

    screen.fill(BLACK)

    """
    This for-loop (and the movement_index) is the only code you would need to copy over/change if you want to add
    another movement, e.g. if you wanted to visualize both the real data and the NN predictions.
    """
    for joint in movement_frames[movement_index]:
        x, y, _ = joint
        pygame.draw.circle(screen, WHITE, [x, y], 2)

    screen.blit(pygame.transform.rotate(screen, 180), (0, 0))
    pygame.display.flip()
    clock.tick(24)

    if movement_index >= len(movement_frames) - 1:
        movement_index = 0
    else:
        movement_index += 1

pygame.quit()