"""
__title__    = New_AR_func.py
__author__   = 'Hansen Wong // 王之超'
__time__     = 2023/3/15 15:37
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

# 定义环境势场
c_sigma = np.array([WIDTH*.5, HEIGHT*.2])


def setting():
    alpha = 0.1
    ATTRACTION_FACTOR = 1
    REPULSION_FACTOR = ATTRACTION_FACTOR * 625
    return alpha, ATTRACTION_FACTOR, REPULSION_FACTOR

def sigma(pos, A_sigma):
    """
    :param pos: The position of the agent in the field
    :param A_sigma: The factor of the Gaussian potential field. If A_sigma > 0, the potential field will have a peak (Divergence). If A_sigma < 0, the potential field will have a valley (Aggregation).
    :return:
    """
    return np.dot(np.array([0, .5]), pos) + A_sigma * 1000 / 2 * np.exp(-(np.linalg.norm(pos - c_sigma)**2) / 10000)


def heading(pos: np.ndarray, A_sigma):
    return -1 * np.array([0, .5]) + \
               A_sigma * 1000 / 10000 * (pos - c_sigma) * np.exp(-(np.linalg.norm(pos - c_sigma)**2) / 10000)

# 定义无人机类
class Drone:
    def __init__(self, pos, A_sigma, alpha: float, ATTRACTION_FACTOR: float, REPULSION_FACTOR: float, start):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array([random.uniform(-MAX_SPEED, MAX_SPEED),
                             random.uniform(-MAX_SPEED, MAX_SPEED)])
        self.max_speed = MAX_SPEED
        self.A_sigma = A_sigma
        self.dis_list = []
        self.history_x = [self.pos[0]]
        self.history_y = [self.pos[1]]
        self.history_vx = [self.vel[0]]
        self.history_vy = [self.vel[1]]
        self.alpha = alpha
        self.ATTRACTION_FACTOR = ATTRACTION_FACTOR
        self.REPULSION_FACTOR = REPULSION_FACTOR
        self.start = start

    # A/R function
    def g(self, pos):
        y = self.pos - pos
        self.dis_list.append(np.linalg.norm(y))
        return -y * (self.ATTRACTION_FACTOR / np.linalg.norm(y) ** 2 - self.REPULSION_FACTOR / np.linalg.norm(y) ** 4)

    # anisotropic factor
    def epsilon(self, pos):
        h = heading(self.pos, self.A_sigma)
        if np.linalg.norm(h) != 0:
            h = h / np.linalg.norm(h)
        else:
            h = np.zeros(2)
        y = pos - self.pos
        y = y / (np.linalg.norm(y))
        return self.alpha * np.dot(h, y)


    def update(self, drones):
        epsilons = np.array([self.epsilon(drone.pos) for drone in drones if
                             drone != self])
        gs = np.array(
            [self.g(drone.pos) for drone in drones if drone != self])
        self.max_dis = max(self.dis_list)
        self.min_dis = min(self.dis_list)
        self.vel = heading(self.pos, self.A_sigma) + np.dot(epsilons, gs)
        if np.linalg.norm(self.vel) > self.max_speed:
            self.vel = self.vel / np.linalg.norm(self.vel) * self.max_speed
        self.vel[0] += random.randint(-10, 10) / 200
        self.vel[1] += random.randint(-10, 10) / 200
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

        # if time.time() - self.start > 15:
        #     self.A_sigma -= 0.1

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


def run(A_sigma):
    alpha, ATTRACTION_FACTOR, REPULSION_FACTOR = setting()

    # 初始化pygame窗口
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    # 初始化字体
    font = pygame.font.SysFont('calibri', 20)

    # 循环更新和绘制无人机
    running = True
    start = time.time()

    # 初始化无人机列表
    drones = [Drone([i*(WIDTH/NUM_DRONES), HEIGHT], A_sigma, alpha, ATTRACTION_FACTOR, REPULSION_FACTOR, start) for i in range(NUM_DRONES)]

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))

        for drone in drones:
            drone.dis_list = []
            drone.update(drones)
            drone.draw(screen)
        potentials = np.array([sigma(drone.pos, drone.A_sigma) for drone in drones])
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

        if time.time() - start > 25:
            running = False

    pygame.quit()

    return drones

def normalization(x, mean, std):
    if type(x) == list:
        for i in range(len(x)):
            x[i] = (x[i] - mean) / std
        return x
    return (x - mean) / std

def denormalization(y, mean, std):
    if type(y) == list:
        for i in range(len(y)):
            y[i] = y[i] * std + mean
        return y
    else:
        return y * std + mean



if __name__ == '__main__':
    run(5)