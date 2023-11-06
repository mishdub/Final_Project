import pygame as pg
from random import randrange
import pymunk.pygame_util
pymunk.pygame_util.positive_y_is_up = False
import time

class Algo6:
    def __init__(self, container_instance, item_instances):
        self.container_instance = container_instance
        self.item_instances = item_instances

    def create_poly(self,space):
        RES = WIDTH, HEIGHT = 800, 600
        # Define the vertices of the polygon (change this according to your desired shape).
        ball_vertices = [(0, 0), (30, 0), (15, 30)]

        # Create a pymunk.Poly object for the ball.
        ball_moment = pymunk.moment_for_poly(1, ball_vertices)
        ball_body = pymunk.Body(1, ball_moment)
        ball_body.position = WIDTH // 2, 0
        ball_shape = pymunk.Poly(ball_body, ball_vertices)
        #ball_shape.friction = 0.1

        # Add the ball to the space.
        space.add(ball_body, ball_shape)

    def plot(self):
        RES = WIDTH, HEIGHT = 800, 600
        FPS = 60
        pg.init()
        surface = pg.display.set_mode(RES)
        clock = pg.time.Clock()
        draw_options = pymunk.pygame_util.DrawOptions(surface)

        space = pymunk.Space()
        space.gravity = 0, 1000

        segment_shape = pymunk.Segment(space.static_body, (0, HEIGHT), (WIDTH, HEIGHT), 20)
        space.add(segment_shape)
        # Create and add multiple falling polygons.
        num_polygons = 5  # Number of polygons to create.delay_between_polygons = 2  # De
        delay_between_polygons = 2  # De

        self.create_poly(space)
            # Ensure the screen is updated during the delay.




        while True:
            surface.fill(pg.Color('black'))

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    return

            space.step(1 / FPS)
            space.debug_draw(draw_options)

            pg.display.flip()
            clock.tick(FPS)