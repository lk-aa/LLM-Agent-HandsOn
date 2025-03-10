import pygame
import random
import sys

# Constants
WINDOW_WIDTH, WINDOW_HEIGHT = 600, 600
GRID_SIZE = 30
GRID_WIDTH, GRID_HEIGHT = WINDOW_WIDTH // GRID_SIZE, WINDOW_HEIGHT // GRID_SIZE
FPS = 15
PACMAN_START_LIVES = 3

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
PINK = (255, 105, 180)
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0)
BLUE = (0, 0, 255)

# Load images
pacman_img = pygame.Surface((GRID_SIZE, GRID_SIZE))
pacman_img.fill(YELLOW)

# Game classes
class Pellet:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, surface):
        pygame.draw.circle(surface, WHITE, (self.x + GRID_SIZE // 2, self.y + GRID_SIZE // 2), 5)

class PowerPellet:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, surface):
        pygame.draw.circle(surface, WHITE, (self.x + GRID_SIZE // 2, self.y + GRID_SIZE // 2), 10)

class Ghost:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.state = 'chase'  # chase, scatter, frightened
        self.color = color

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, (self.x, self.y, GRID_SIZE, GRID_SIZE))

    def move(self, pacman_x, pacman_y):
        if self.state == 'chase':
            if pacman_x > self.x:
                self.x += GRID_SIZE
            elif pacman_x < self.x:
                self.x -= GRID_SIZE
            if pacman_y > self.y:
                self.y += GRID_SIZE
            elif pacman_y < self.y:
                self.y -= GRID_SIZE

class PacMan:
    def __init__(self):
        self.x = GRID_SIZE
        self.y = GRID_SIZE
        self.lives = PACMAN_START_LIVES
        self.direction = (1, 0)  # Moving right
        self.powered = False
        self.power_timer = 0

    def move(self):
        self.x += GRID_SIZE * self.direction[0]
        self.y += GRID_SIZE * self.direction[1]

        # Wrap around screen
        if self.x >= WINDOW_WIDTH:
            self.x = 0
        if self.y >= WINDOW_HEIGHT:
            self.y = 0
        if self.x < 0:
            self.x = WINDOW_WIDTH - GRID_SIZE
        if self.y < 0:
            self.y = WINDOW_HEIGHT - GRID_SIZE

    def change_direction(self, new_direction):
        self.direction = new_direction

# Game initialization
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Pac-Man")
clock = pygame.time.Clock()

# Game state
pellets = [Pellet(x * GRID_SIZE, y * GRID_SIZE) for x in range(GRID_WIDTH) for y in range(GRID_HEIGHT)]
power_pellets = [PowerPellet(0, 0), PowerPellet(WINDOW_WIDTH - GRID_SIZE, 0), PowerPellet(0, WINDOW_HEIGHT - GRID_SIZE), PowerPellet(WINDOW_WIDTH - GRID_SIZE, WINDOW_HEIGHT - GRID_SIZE)]
ghosts = [Ghost(3 * GRID_SIZE, 3 * GRID_SIZE, RED),
          Ghost(3 * GRID_SIZE, 4 * GRID_SIZE, PINK),
          Ghost(3 * GRID_SIZE, 5 * GRID_SIZE, CYAN),
          Ghost(3 * GRID_SIZE, 6 * GRID_SIZE, ORANGE)]

pacman = PacMan()

# Main loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        pacman.change_direction((-1, 0))
    elif keys[pygame.K_RIGHT]:
        pacman.change_direction((1, 0))
    elif keys[pygame.K_UP]:
        pacman.change_direction((0, -1))
    elif keys[pygame.K_DOWN]:
        pacman.change_direction((0, 1))

    pacman.move()

    for pellet in pellets[:]:
        if pacman.x == pellet.x and pacman.y == pellet.y:
            pellets.remove(pellet)

    for power_pellet in power_pellets[:]:
        if pacman.x == power_pellet.x and pacman.y == power_pellet.y:
            power_pellets.remove(power_pellet)
            pacman.powered = True
            pacman.power_timer = 50  # Duration of power

    if pacman.powered:
        pacman.power_timer -= 1
        if pacman.power_timer <= 0:
            pacman.powered = False

    for ghost in ghosts:
        ghost.move(pacman.x, pacman.y)

    # Clear the screen
    screen.fill(BLACK)

    # Draw game elements
    for pellet in pellets:
        pellet.draw(screen)
    for power_pellet in power_pellets:
        power_pellet.draw(screen)
    for ghost in ghosts:
        ghost.draw(screen)
    screen.blit(pacman_img, (pacman.x, pacman.y))

    pygame.display.flip()
    clock.tick(FPS)
