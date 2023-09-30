"""
__title__    = boids.py
__author__   = 'Hansen Wong // 王之超'
__time__     = 2023/2/20 13:25
__Software__ = Pycharm
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
             ┏┓    ┏┓
            ┏┛┻━━━┛ ┻┓
            ┃         ┃
            ┃  ┳┛  ┗┳  ┃
            ┃     ┻    ┃
            ┗━┓      ┏━┛
              ┃      ┗━━━┓
              ┃  神兽保佑  ┣┓
              ┃　永无BUG！ ┏┛
                ┗┓┓┏━┳┓┏┛
                 ┃┫┫  ┃┫┫
                 ┗┻┛  ┗┻┛
"""
import numpy as np
import pygame
import random

# 定义窗口大小
WIDTH = 800
HEIGHT = 600

# 定义无人机颜色和大小
DRONE_COLOR = (255, 255, 255)
DRONE_SIZE = 5

# 定义Boids fleet参数
NUM_DRONES = 20
MAX_SPEED = 1
ALIGNMENT_FACTOR = 20
COHESION_FACTOR = 1
SEPARATION_FACTOR = 100

# 初始化pygame窗口
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))

c_sigma = np.array([WIDTH * .5, HEIGHT * .5])
A_sigma = 1
def heading(pos: np.ndarray):
    return -A_sigma * (pos - c_sigma)
    # h1 = 20000 / 5000 * (pos - c_sigma) * np.exp(-np.linalg.norm(pos - c_sigma)**2 / 5000)
    # h2 = -4000 / 5000 * (pos - np.array([400, 400])) * np.exp(-np.linalg.norm(pos - np.array([400, 400]))**2 / 5000)
    # h3 = -4000 / 5000 * (pos - np.array([400, 200])) * np.exp(-np.linalg.norm(pos - np.array([400, 200])) ** 2 / 5000)
    # h4 = -4000 / 5000 * (pos - np.array([300, 300])) * np.exp(-np.linalg.norm(pos - np.array([300, 300])) ** 2 / 5000)
    # h5 = -4000 / 5000 * (pos - np.array([500, 300])) * np.exp(-np.linalg.norm(pos - np.array([500, 300])) ** 2 / 5000)
    # h6 = -.2 * (pos - c_sigma)
    # return h1 + h2 + h3 + h4 + h5 + h6

# 定义无人机类
class Drone:
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array([random.uniform(-MAX_SPEED, MAX_SPEED),
                             random.uniform(-MAX_SPEED, MAX_SPEED)])
        self.max_speed = MAX_SPEED

    def update(self, drones):
        alignment = np.mean([drone.vel for drone in drones if drone != self], axis=0)
        cohesion = np.mean([drone.pos for drone in drones if drone != self], axis=0) - self.pos
        separation = np.zeros(2)
        close_drones = [drone for drone in drones if drone != self and np.linalg.norm(self.pos - drone.pos) < 50]
        if len(close_drones) > 0:
            separation = np.mean([self.pos - drone.pos for drone in close_drones], axis=0)
            if np.linalg.norm(separation) > 0:
                separation = separation / np.linalg.norm(separation)
        alignment *= ALIGNMENT_FACTOR
        cohesion *= COHESION_FACTOR
        separation *= SEPARATION_FACTOR
        # self.vel += alignment + cohesion + separation
        self.vel += alignment + cohesion + separation + heading(self.pos)
        if np.linalg.norm(self.vel) > self.max_speed:
            self.vel = self.vel / np.linalg.norm(self.vel) * self.max_speed
        self.pos += self.vel

    def draw(self):
        pos = tuple(self.pos.astype(int))
        pygame.draw.circle(screen, DRONE_COLOR, pos, DRONE_SIZE)


# 初始化无人机列表
drones = [Drone([random.randint(0, WIDTH), random.randint(0, HEIGHT)]) for i in range(NUM_DRONES)]

# 循环更新和绘制无人机
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 0))

    for drone in drones:
        drone.update(drones)
        drone.draw()

    pygame.display.flip()

pygame.quit()
