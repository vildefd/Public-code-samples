#!/usr/bin/env python
import pygame
from pygame import Vector2
import random

class MovingObject:
    def __init__(self, screen_res, init_position = Vector2(42, 200), init_size = 30, init_speed=None, init_life = 1):
        if not init_speed:
            self.speed = Vector2(50 + 60 * random.random(), 50 + 60 * random.random())
        else:
            self.speed = init_speed
        self.size = init_size
        self.pos = init_position
        self.event = False
        self.life = init_life
        self._alive = True

    def move(self, time_passed, screen_res):
        self.pos.x += self.speed.x * time_passed
        self.pos.y += self.speed.y * time_passed
        
        if self.pos.x < 0:
            self.speed.x = abs(self.speed.x)
        if self.pos.y < 0:
            self.speed.y = abs(self.speed.y)

        if self.pos.x > screen_res[0] - self.size:
            self.speed.x = -abs(self.speed.x)
        if self.pos.y > screen_res[1] - self.size:
            self.speed.y = -abs(self.speed.y)
    
    def lose_life(self):
        if abs(self.life) > 0:
            self.life = abs(self.life) - 1
            print("Remaining life: {}".format(self.life))
        else:
            self._alive = False

    def is_alive(self):
        return self._alive


class Ball(MovingObject):
    def __init__(self, screen_res, ball_img, lifes = 1):
        MovingObject.__init__(self, screen_res, init_life = lifes)
        self.img = ball_img
        self.size = self.img.get_height()

    def move(self, time_passed, screen_res):
        self.pos.x += self.speed.x * time_passed
        self.pos.y += self.speed.y * time_passed
        
        if self.pos.x < 0:
            self.speed.x = abs(self.speed.x)
        
        if self.pos.y < 0:
            self.speed.y = abs(self.speed.y)

        if self.pos.x > screen_res[0] - self.size:
            self.speed.x = -abs(self.speed.x)

    def inspect(self, boundary):
        if self.pos.y > boundary[1] - self.size:
            self.lose_life()
            self.speed.y = -abs(self.speed.y)

    def draw(self, screen):
        screen.blit(self.img, (self.pos.x, self.pos.y))


class Mirror():
    def __init__(self, x, y, width, height):
        self.pos = Vector2(x, y)        
        self.col = (0, 255, 0)
        self.width = width
        self.height = height
    
    def move(self, x):
        self.pos.x = x

    def draw(self, screen):
        pygame.draw.rect(screen, self.col, 
                        ( round(self.pos.x), 
                        round(self.pos.y) , 
                        round(self.width), 
                        round(self.height) ))

class Balloon():
    def __init__(self, x, y, width, height, color = (125, 125, 125), init_life = 1):
        self.col = color
        self.pos = Vector2(x, y)
        self.width = width
        self.height = height
        self.render = True
        self.life = abs(init_life)
    
    def draw(self, screen):
        pygame.draw.rect(screen, self.col, 
                        ( round(self.pos.x), 
                        round(self.pos.y) , 
                        round(self.width), 
                        round(self.height) ))

    