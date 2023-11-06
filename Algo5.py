
import pygame
import random
import math

class Algo4:
    def __init__(self, container_instance, item_instances):
        self.container_instance = container_instance
        self.item_instances = item_instances

    def plot(self):
        # Initialize pygame
        pygame.init()

        # Constants
        SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
        PARTICLE_RADIUS = 5
        GRAVITY = 0.5
        BOUNCE_FACTOR = -0.8  # Adjusted bounce factor
        TIME_STEP = 0.5

        # Colors
        WHITE = (255, 255, 255)
        BLUE = (0, 0, 255)

        # Particle class
        class Particle:
            def __init__(self, x, y):
                self.x = x
                self.y = y
                self.y_velocity = 0

            def apply_gravity(self):
                self.y_velocity += GRAVITY
                self.y += self.y_velocity * TIME_STEP

                # Bounce off the ground
                if self.y > SCREEN_HEIGHT:
                    self.y = SCREEN_HEIGHT
                    self.y_velocity *= BOUNCE_FACTOR

        # Initialize the screen
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Particle Simulation")

        # Create a list of particles
        particles = [Particle(random.randint(100, SCREEN_WIDTH - 100), random.randint(50, 200)) for _ in range(100)]

        # Main loop
        running = True
        clock = pygame.time.Clock()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.fill(WHITE)

            # Apply gravity to the particles
            for particle in particles:
                particle.apply_gravity()

            # Draw the particles
            for particle in particles:
                pygame.draw.circle(screen, BLUE, (int(particle.x), int(particle.y)), PARTICLE_RADIUS)

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()

    def plot2(self):
        # Initialize pygame
        pygame.init()

        # Constants
        SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
        TRIANGLE_SIZE = 20
        GRAVITY = 0.5
        BOUNCE_FACTOR = 0
        TIME_STEP = 0.8
        MIN_DISTANCE = TRIANGLE_SIZE * 2  # Minimum distance to maintain between triangles

        # Colors
        WHITE = (255, 255, 255)
        BLUE = (0, 0, 255)

        # Particle class
        class Particle:
            def __init__(self, x, y):
                self.x = x
                self.y = y
                self.y_velocity = 0

            # Add a damping constant
            DAMPING = 0.98  # Adjust this value to control the damping effect

            # Inside the Particle class's apply_gravity method, update it as follows:
            def apply_gravity(self):
                self.y_velocity += GRAVITY
                self.y_velocity *= 0.8  # Apply damping
                self.y += self.y_velocity * TIME_STEP

                # Bound the triangles to the ground level
                if self.y > SCREEN_HEIGHT - TRIANGLE_SIZE:  # Adjust for triangle size
                    self.y = SCREEN_HEIGHT - TRIANGLE_SIZE
                    self.y_velocity *= BOUNCE_FACTOR

                if self.x < TRIANGLE_SIZE:
                    self.x = TRIANGLE_SIZE
                elif self.x > SCREEN_WIDTH - TRIANGLE_SIZE:
                    self.x = SCREEN_WIDTH - TRIANGLE_SIZE

            def check_collision(self, other_particle):
                dx = other_particle.x - self.x
                dy = other_particle.y - self.y
                distance = math.sqrt(dx * dx + dy * dy)
                return distance < MIN_DISTANCE

        # Initialize the screen
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Particle Simulation")

        # Create a list of particles (triangles)
        particles = []
        for _ in range(100):
            x = random.randint(0, SCREEN_WIDTH - 2 * TRIANGLE_SIZE) + TRIANGLE_SIZE
            y = random.randint(0, SCREEN_HEIGHT - TRIANGLE_SIZE)
            particles.append(Particle(x, y))

        # Main loop
        running = True
        clock = pygame.time.Clock()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.fill(WHITE)

            # Apply gravity to the particles
            for particle in particles:
                particle.apply_gravity()

            # Check for collisions and resolve them
            for i in range(len(particles)):
                for j in range(i + 1, len(particles)):
                    if particles[i].check_collision(particles[j]):
                        # Implement collision response logic here
                        # Adjust positions to prevent overlap
                        dx = particles[j].x - particles[i].x
                        dy = particles[j].y - particles[i].y
                        distance = math.sqrt(dx * dx + dy * dy)
                        overlap = (MIN_DISTANCE - distance) / 2

                        # Move particles in opposite directions
                        angle = math.atan2(dy, dx)
                        particles[i].x -= overlap * math.cos(angle)
                        particles[i].y -= overlap * math.sin(angle)
                        particles[j].x += overlap * math.cos(angle)
                        particles[j].y += overlap * math.sin(angle)

            # Draw the particles as triangles
            for particle in particles:
                x, y = int(particle.x), int(particle.y)
                size = TRIANGLE_SIZE
                pygame.draw.polygon(screen, BLUE, [(x, y), (x + size, y), (x + size / 2, y - size)], 0)

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()

    def plot3(self):
        # Initialize pygame
        pygame.init()

        # Constants
        SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
        TRIANGLE_SIZE = 20
        GRAVITY = 0.5
        BOUNCE_FACTOR = 0
        TIME_STEP = 0.8
        MIN_DISTANCE = TRIANGLE_SIZE * 2  # Minimum distance to maintain between triangles
        ADD_INTERVAL = 60  # Interval to add triangles (adjust as needed)

        # Colors
        WHITE = (255, 255, 255)
        BLUE = (0, 0, 255)

        # Particle class
        class Particle:
            def __init__(self, x, y):
                self.x = x
                self.y = y
                self.y_velocity = 0

            # Add a damping constant
            DAMPING = 0.98  # Adjust this value to control the damping effect

            # Inside the Particle class's apply_gravity method, update it as follows:
            def apply_gravity(self):
                self.y_velocity += GRAVITY
                self.y_velocity *= 0.8  # Apply damping
                self.y += self.y_velocity * TIME_STEP

                # Bound the triangles to the ground level
                if self.y > SCREEN_HEIGHT - TRIANGLE_SIZE:  # Adjust for triangle size
                    self.y = SCREEN_HEIGHT - TRIANGLE_SIZE
                    self.y_velocity *= BOUNCE_FACTOR

                if self.x < TRIANGLE_SIZE:
                    self.x = TRIANGLE_SIZE
                elif self.x > SCREEN_WIDTH - TRIANGLE_SIZE:
                    self.x = SCREEN_WIDTH - TRIANGLE_SIZE

            def check_collision(self, other_particle):
                dx = other_particle.x - self.x
                dy = other_particle.y - self.y
                distance = math.sqrt(dx * dx + dy * dy)
                return distance < MIN_DISTANCE

        # Initialize the screen
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Particle Simulation")

        # Create a list of particles (triangles)
        particles = []

        # Main loop
        running = True
        clock = pygame.time.Clock()
        frame_counter = 0  # Count frames for adding particles
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.fill(WHITE)

            # Apply gravity to the particles
            for particle in particles:
                particle.apply_gravity()

            # Check for collisions and resolve them
            for i in range(len(particles)):
                for j in range(i + 1, len(particles)):
                    if particles[i].check_collision(particles[j]):
                        # Implement collision response logic here
                        # Adjust positions to prevent overlap
                        dx = particles[j].x - particles[i].x
                        dy = particles[j].y - particles[i].y
                        distance = math.sqrt(dx * dx + dy * dy)
                        overlap = (MIN_DISTANCE - distance) / 2

                        # Move particles in opposite directions
                        angle = math.atan2(dy, dx)
                        particles[i].x -= overlap * math.cos(angle)
                        particles[i].y -= overlap * math.sin(angle)
                        particles[j].x += overlap * math.cos(angle)
                        particles[j].y += overlap * math.sin(angle)

            # Add a new particle from the middle of the screen at the defined interval
            frame_counter += 1
            if frame_counter == ADD_INTERVAL:
                x = SCREEN_WIDTH // 2  # Middle of the screen
                y = 0
                particles.append(Particle(x, y))
                frame_counter = 0  # Reset frame counter

            # Draw the particles as triangles
            for particle in particles:
                x, y = int(particle.x), int(particle.y)
                size = TRIANGLE_SIZE
                pygame.draw.polygon(screen, BLUE, [(x, y), (x + size, y), (x + size / 2, y - size)], 0)

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()

    def plot4(self):
        # Initialize pygame
        pygame.init()

        # Constants
        SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
        TRIANGLE_SIZE = 20
        GRAVITY = 0.5
        BOUNCE_FACTOR = 0
        TIME_STEP = 0.8
        MIN_DISTANCE = TRIANGLE_SIZE * 2
        ADD_INTERVAL = 60

        # Colors
        WHITE = (255, 255, 255)
        BLUE = (0, 0, 255)

        # Convex region coordinates (modify these to create your convex shape)
        convex_region = [
            (200, 200),
            (300, 100),
            (500, 100),
            (600, 200),
        ]

        # Particle class
        class Particle:
            def __init__(self, x, y):
                self.x = x
                self.y = y
                self.y_velocity = 0

            DAMPING = 0.98

            def apply_gravity(self):
                self.y_velocity += GRAVITY
                self.y_velocity *= 0.8
                self.y += self.y_velocity * TIME_STEP

                # Bound the triangles within the convex region
                for i in range(len(convex_region)):
                    x1, y1 = convex_region[i]
                    x2, y2 = convex_region[(i + 1) % len(convex_region)]
                    if (x1 - self.x) * (y2 - self.y) - (x2 - self.x) * (y1 - self.y) > 0:
                        # Point is outside the convex region, perform collision response
                        dx = self.x - x1
                        dy = self.y - y1
                        edge_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        overlap = (dx * (x2 - x1) + dy * (y2 - y1)) / edge_length ** 2
                        overlap = min(max(overlap, 0), edge_length)
                        nearest_x = x1 + overlap * (x2 - x1) / edge_length
                        nearest_y = y1 + overlap * (y2 - y1) / edge_length

                        distance = math.sqrt((self.x - nearest_x) ** 2 + (self.y - nearest_y) ** 2)
                        if distance < TRIANGLE_SIZE:
                            # Collision detected, adjust position
                            angle = math.atan2(nearest_y - self.y, nearest_x - self.x)
                            self.x = nearest_x - TRIANGLE_SIZE * math.cos(angle)
                            self.y = nearest_y - TRIANGLE_SIZE * math.sin(angle)

            def check_collision(self, other_particle):
                dx = other_particle.x - self.x
                dy = other_particle.y - self.y
                distance = math.sqrt(dx * dx + dy * dy)
                return distance < MIN_DISTANCE

        # Initialize the screen
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Particle Simulation")

        # Create a list of particles (triangles)
        particles = []

        # Main loop
        running = True
        clock = pygame.time.Clock()
        frame_counter = 0
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.fill(WHITE)

            for particle in particles:
                particle.apply_gravity()

            for i in range(len(particles)):
                for j in range(i + 1, len(particles)):
                    if particles[i].check_collision(particles[j]):
                        dx = particles[j].x - particles[i].x
                        dy = particles[j].y - particles[i].y
                        distance = math.sqrt(dx * dx + dy * dy)
                        overlap = (MIN_DISTANCE - distance) / 2

                        angle = math.atan2(dy, dx)
                        particles[i].x -= overlap * math.cos(angle)
                        particles[i].y -= overlap * math.sin(angle)
                        particles[j].x += overlap * math.cos(angle)
                        particles[j].y += overlap * math.sin(angle)

            frame_counter += 1
            if frame_counter == ADD_INTERVAL:
                # Add a new particle from the middle of the convex region at the defined interval
                x = sum(point[0] for point in convex_region) / len(convex_region)
                y = sum(point[1] for point in convex_region) / len(convex_region)
                particles.append(Particle(x, y))
                frame_counter = 0

            # Draw the particles as triangles
            for particle in particles:
                x, y = int(particle.x), int(particle.y)
                size = TRIANGLE_SIZE
                pygame.draw.polygon(screen, BLUE, [(x, y), (x + size, y), (x + size / 2, y - size)], 0)

            # Draw the convex region
            pygame.draw.polygon(screen, (0, 255, 0), convex_region, 1)

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()
