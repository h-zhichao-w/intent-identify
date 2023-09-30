"""
__title__    = BoidsDroneDefine.py
__author__   = 'Hansen Wong // 王之超'
__time__     = 2023/2/20 13:46
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
import time
import random
import pandas as pd

# 定义窗口大小
WIDTH = 500
HEIGHT = 500

# 定义无人机颜色和大小
DRONE_COLOR = (255, 255, 255)
DRONE_SIZE = 5

# 定义Boids fleet参数
NUM_DRONES = 20
MAX_SPEED = 1
ATTRACTION_FACTOR = .1
REPULSION_FACTOR = 200

# 定义环境势场
# A_sigma = 1
c_sigma = np.array([WIDTH*.5, HEIGHT*.2])
alpha = 0.1


def setting(mode: str):
    if mode == 'Alignment':
        alpha = .1
        ATTRACTION_FACTOR = .1
        REPULSION_FACTOR = 500

    elif mode == 'Aggregation':
        alpha = .1
        ATTRACTION_FACTOR = .1
        REPULSION_FACTOR = 200

    elif mode == 'Divergence':
        alpha = .38
        ATTRACTION_FACTOR = .08
        REPULSION_FACTOR = 500


def sigma(pos, mode: str):
    # Plane (Alignment)
    if mode == 'Alignment':
        return np.dot(np.array([0, .5]), pos)

    # Plane_Gaussian (Aggregation)
    elif mode == 'Aggregation':
        return np.dot(np.array([0, .5]), pos) - 1000 / 2 * np.exp(-(np.linalg.norm(pos - c_sigma) ** 2) / 10000)

    # Plane_Gaussian (Divergence)
    elif mode == 'Divergence':
        return np.dot(np.array([0, .5]), pos) + 1000 / 2 * np.exp(-(np.linalg.norm(pos - c_sigma)**2) / 10000)

    # Gaussian
    # return -100/2 * np.exp(-(np.linalg.norm(pos - np.array([250, 100]))**2) / 10000)

    # Quadratic
    # return A_sigma / 2 * np.linalg.norm(pos - c_sigma)**2


def heading(pos: np.ndarray, mode: str):
    # Gaussian
    # return -800 / 1000000 * (pos - np.array([250, 100])) * np.exp(-(np.linalg.norm(pos - np.array([250, 100]))**2) / 10000)

    # Plane (Alignment)
    if mode == 'Alignment':
        return -1 * np.array([0, .5])

    # Quadratic
    # return -A_sigma * (pos - c_sigma)

    # Plane_Gaussian (Divergence)
    elif mode == 'Divergence':
        return -1 * np.array([0, .5]) + \
               1000 / 10000 * (pos - c_sigma) * np.exp(-(np.linalg.norm(pos - c_sigma)**2) / 10000)

    # Plane_Gaussian (Aggregation)
    elif mode == 'Aggregation':
        return -1 * np.array([0, .5]) - \
               1000 / 10000 * (pos - c_sigma) * np.exp(-(np.linalg.norm(pos - c_sigma)**2) / 10000)

    # Multimodal_Gaussian_and_Quadratic
    # h1 = 20000 / 5000 * (pos - c_sigma) * np.exp(-np.linalg.norm(pos - c_sigma)**2 / 5000)
    # h2 = -4000 / 5000 * (pos - np.array([400, 400])) * np.exp(-np.linalg.norm(pos - np.array([400, 400]))**2 / 5000)
    # h3 = -4000 / 5000 * (pos - np.array([400, 200])) * np.exp(-np.linalg.norm(pos - np.array([400, 200])) ** 2 / 5000)
    # h4 = -4000 / 5000 * (pos - np.array([300, 300])) * np.exp(-np.linalg.norm(pos - np.array([300, 300])) ** 2 / 5000)
    # h5 = -4000 / 5000 * (pos - np.array([500, 300])) * np.exp(-np.linalg.norm(pos - np.array([500, 300])) ** 2 / 5000)
    # h6 = -.2 * (pos - c_sigma)
    # return h1 + h2 + h3 + h4 + h5 + h6

def noise(true_value):
    n = .5 * np.sin(random.randint(-10, 10) / 10 * 0.5 * np.pi)
    return true_value + n


# 定义无人机类
class Drone:
    def __init__(self, pos, mode: str):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array([random.uniform(-MAX_SPEED, MAX_SPEED),
                             random.uniform(-MAX_SPEED, MAX_SPEED)])
        self.max_speed = MAX_SPEED
        self.mode = mode
        self.dis_list = []
        self.history_x = [self.pos[0]]
        self.history_y = [self.pos[1]]
        self.history_vx = [self.vel[0]]
        self.history_vy = [self.vel[1]]

    # A/R function
    def g(self, pos):
        y = self.pos - pos
        self.dis_list.append(np.linalg.norm(y))
        return -y * (ATTRACTION_FACTOR - REPULSION_FACTOR / np.linalg.norm(y)**2)

    # anisotropic factor
    def epsilon(self, pos):
        h = heading(self.pos, self.mode)
        if np.linalg.norm(h) != 0:
            h = h / np.linalg.norm(h)
        else:
            h = np.zeros(2)
        y = pos - self.pos
        y = y / (np.linalg.norm(y))
        return alpha * np.dot(h, y)


    def update(self, drones):
        epsilons = np.array([self.epsilon(drone.pos) for drone in drones if
                             drone != self])
        gs = np.array(
            [self.g(drone.pos) for drone in drones if drone != self])
        self.max_dis = max(self.dis_list)
        self.min_dis = min(self.dis_list)
        # self.vel = heading(self.pos) + np.sum(epsilons[:, np.newaxis] * gs, axis=0)
        self.vel = heading(self.pos, self.mode) + np.dot(epsilons, gs)
        if np.linalg.norm(self.vel) > self.max_speed:
            self.vel = self.vel / np.linalg.norm(self.vel) * self.max_speed
        self.pos += self.vel

        # 处理边界
        if self.pos[0] < 0:
            self.pos[0] = 0
            self.vel[0] *= 0
        elif self.pos[0] > WIDTH:
            self.pos[0] = WIDTH
            self.vel[0] *= 0
        if self.pos[1] < 0:
            self.pos[1] = 0
            self.vel[1] *= 0
        elif self.pos[1] > HEIGHT:
            self.pos[1] = HEIGHT
            self.vel[1] *= 0

        # 噪声
        # self.pos[0] = noise(self.pos[0])
        # self.pos[1] = noise(self.pos[1])

        # 保存数据
        self.history_x.append(self.pos[0])
        self.history_y.append(self.pos[1])
        self.history_vx.append(self.vel[0])
        self.history_vy.append(self.vel[1])

    def draw(self, screen):
        pos1 = tuple(self.pos.astype(int) + np.array([3, 3]))
        pos2 = tuple(self.pos.astype(int) + np.array([-3, 3]))
        pos = tuple(self.pos.astype(int))
        pygame.draw.polygon(screen, DRONE_COLOR, [pos, pos1, pos2])


def run(mode: str):
    setting(mode)

    # 初始化无人机列表
    drones = [Drone([i*(WIDTH/NUM_DRONES), HEIGHT], mode) for i in range(NUM_DRONES)]

    # 初始化pygame窗口
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    # 初始化字体
    font = pygame.font.SysFont('calibri', 20)

    # 循环更新和绘制无人机
    running = True
    start = time.time()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))

        for drone in drones:
            drone.dis_list = []
            drone.update(drones)
            drone.draw(screen)
        potentials = np.array([sigma(drone.pos, drone.mode) for drone in drones])
        potential = np.mean(potentials)
        text = font.render('Average potential:{:.2f}'.format(potential), True, (255, 255, 255))
        screen.blit(text, (0, 0))

        max_dis = max(np.array([drone.max_dis for drone in drones]))
        text = font.render('Max inter-distance:{:.2f}'.format(max_dis), True, (255, 255, 255))
        screen.blit(text, (0, 25))

        max_dis = min(np.array([drone.min_dis for drone in drones]))
        text = font.render('Min inter-distance:{:.2f}'.format(max_dis), True, (255, 255, 255))
        screen.blit(text, (0, 50))

        pygame.display.flip()

        if time.time() - start > 20:
            running = False

    pygame.quit()

    return drones

def normalization(x, mean, std):
    if type(x) == list:
        for i in range(len(x)):
            x[i] = (x[i] - mean) / std
        return x
    else:
        return (x - mean) / std

def denormalization(y, mean, std):
    if type(y) == list:
        for i in range(len(y)):
            y[i] = y[i] * std + mean
        return y
    else:
        return y * std + mean

if __name__ == '__main__':
    run('Alignment')