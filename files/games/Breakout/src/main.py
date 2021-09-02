#!/usr/bin/env python3

import pygame 
from pygame import Vector2
import random 
import numpy as np
import vector as vec
import movingobjects
from movingobjects import Mirror, Ball, Balloon
import winsound

#Initialize
screen_res = (640, 480)
BG_FNAME = "img/bg_img.png"
BALL_FNAME = "img/ball.png"

pygame.init()
pygame.display.set_caption("Breakout - vdr004") #set window title
screen = pygame.display.set_mode(screen_res, 0, 32)
background = pygame.image.load(BG_FNAME).convert()

ball_img = pygame.image.load(BALL_FNAME).convert_alpha()

# The moving ball
ball = Ball(screen_res, ball_img, 2)
ball_rad =  ball.size//2

#The "mirror" to move around
mirror_width = 100
mirror_height = 20
mirror = Mirror(screen_res[0]//2 - mirror_width//2, screen_res[1] - mirror_height,
                mirror_width, mirror_height)

#Create rectangular "balloons"
rect_width = 40
rect_height = 20

num_object_height = 8 #number of balloons along height
num_object_width =  screen_res[0]//rect_width #number of balloons along width
num_objects = num_object_width * num_object_height #total number of balloons

objs = []
for i in range(num_object_height):
    for j in range(num_object_width):
        objs.append(Balloon(0 + j*rect_width, 0 + i*rect_height, rect_width, rect_height))
        objs[-1].col = ( random.randint(150, 225), random.randint(150, 225), random.randint(150, 225) )

#print(np.shape(objs))

def clean_object_list(inputlist):
    # inputlist - list of objects, with the property 'render' indicating whether or not to remove
    #
    # Removes one element from list, then checks if there are other, 
    # returns new list of objects 
    remove_this = []  
    tmp = inputlist #temporary list

    if tmp:
        #Check if there are any elements to remove, 
        # continue to while loop at first occurence
        for i in range( len(tmp) ):
            if(tmp[i].render == False):
                remove_this.append(i)
                break
        
        # Repeatedly check for any elements to remove
        # until there are none marked for removal left
        while remove_this and tmp:
            del tmp[remove_this[0]]
            remove_this.clear()
            for i in range( len(tmp) ):
                if not tmp[i].render:
                    remove_this.append(i)
                    break    
    return tmp

#The "main" function, where everything dynamic happens:
clock = pygame.time.Clock()
while True:
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            exit(0)

    time_passed = clock.tick(30) / 1000.0

    x, y = pygame.mouse.get_pos()

    screen.blit(background, (0, 0))
    
    if x >= 0 and x <= screen_res[0]:
        mirror.move(x - mirror_width//2)
    
    mirror.draw(screen)

    #Check if the mirror and the ball intersect
    mirror_impact = vec.intersect_rectangle_circle(mirror.pos, mirror.width, mirror.height,
                                                    ball.pos, ball.size/2, ball.speed)

    if mirror_impact:
        mirror_rad = mirror_width / 2 # "radius" of a circular mirror
        mirror_center = (mirror.pos.x + mirror.width//2, mirror.pos.y + mirror.height//2)
        direction = vec.intersect_circles(mirror_center, mirror_rad, ball.pos, ball.size/2)
        if direction:
            ball.speed.x = 110*round(direction.x)
            ball.speed.y = 110*round(direction.y) - 10
        #print(direction)

    #Check if ball intersect with "balloon" object, "pop" if intersected
    bounce_back = False
    for i in range(len(objs)):
        impact = vec.intersect_rectangle_circle(objs[i].pos, objs[i].width, objs[i].height,
                                                    ball.pos, ball.size//2, ball.speed)
        if impact:
            objs[i].render = False
            bounce_back = True
                
        if objs[i].render:
            objs[i].draw(screen)

    if bounce_back:
        ball.speed.y = (-1)*ball.speed.y
        winsound.Beep(3000, 200)

    ball.move(time_passed, screen_res)
    ball.draw(screen)
    ball.inspect(screen_res)

    pygame.display.update()

    #Clean up:
    objs = clean_object_list(objs)

    #Check if you've "won" or "lost"
    if not ball.is_alive():
        print("Oh no! You lost the ball!")
        winsound.Beep(900, 500)
        winsound.Beep(640, 500)
        winsound.Beep(240, 1000)
        pygame.quit()
        exit(0)

    if not objs:
        print("Congratulations! You did it!")
        winsound.Beep(440, 500)
        winsound.Beep(640, 500)
        winsound.Beep(1200, 1000)        
        pygame.quit()
        exit(0)


