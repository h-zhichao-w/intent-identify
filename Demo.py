"""
__title__    = Demo.py
__author__   = 'Hansen Wong // 王之超'
__time__     = 2023/3/9 18:04
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
import time
from New_AR_func import *
from keras import models
# import cv2

data_mean = 106.03646163873081
data_std = 151.59025900983642

route_model = models.load_model('dataset_new/best_model.h5')
intention_model = models.load_model('dataset_new/best_model_intent.h5')
INPUT = []
PREDICT_COL = (255, 0, 0)

A_sigma = -5

alpha, ATTRACTION_FACTOR, REPULSION_FACTOR = setting()

# 初始化无人机列表
drones = [Drone([i*(WIDTH/NUM_DRONES), HEIGHT], A_sigma, alpha, ATTRACTION_FACTOR, REPULSION_FACTOR, None) for i in range(NUM_DRONES)]

# 初始化pygame窗口
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Intention Recognition')

# 创建视频编码器
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# video_writer = cv2.VideoWriter('Demo/Alignment.mp4', fourcc, 60, (WIDTH, HEIGHT))

# 初始化字体
font = pygame.font.SysFont('calibri', 20)

# 循环更新和绘制无人机
running = True
start = time.time()
step = 0
Aggregation = []
Alignment = []
Divergence = []
Time = []
Prediction = []
Coor_x, Coor_y = [], []
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 0))

    data = [np.cos(step / 100 * np.pi), np.sin(step / 100 * np.pi)]
    for drone in drones:
        drone.dis_list = []
        drone.update(drones)
        drone.draw(screen)
        data.extend(normalization([drone.pos[0], drone.pos[1], drone.vel[0], drone.vel[1]], data_mean, data_std))

    INPUT.append(data)
    if len(INPUT) == 5:
        X = np.array(INPUT).reshape((1, 5, 82))
        prediction = route_model.predict(X)
        y_predict = prediction.reshape((40, ))
        OUTPUT = denormalization(y_predict, data_mean, data_std)
        Prediction.append(OUTPUT)
        # print(OUTPUT)
        INPUT.pop(0)
        intention = intention_model.predict(prediction.reshape((1, 1, 40))).reshape((3, ))
        # print(intention)
        for i in range(20):
            center = (float(OUTPUT[i*2]), float(OUTPUT[i*2+1]))
            # print(type(OUTPUT[i*2]))
            pygame.draw.circle(screen, PREDICT_COL, center, 3)
        aggregation = font.render('Aggregation:{:.2f}'.format(intention[0]), True, (255, 255, 255))
        alignment = font.render('Alignment:{:.2f}'.format(intention[1]), True, (255, 255, 255))
        divergence = font.render('Divergence:{:.2f}'.format(intention[2]), True, (255, 255, 255))
        screen.blit(aggregation, (350, 0))
        screen.blit(alignment, (350, 25))
        screen.blit(divergence, (350, 50))
        Aggregation.append(intention[0])
        Alignment.append(intention[1])
        Divergence.append(intention[2])
        t = time.time() - start
        Time.append(t)
    else:
        pass

    T = font.render('Time:{:.2f}'.format(time.time() - start), True, (255, 255, 255))
    screen.blit(T, (0, 75))

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
    step += 1

    # 获取Pygame窗口截图
    # surface = pygame.display.get_surface()
    # image = pygame.surfarray.array3d(surface)
    # image = np.transpose(image, (1, 0, 2))  # 转置图像数组，使它与OpenCV默认的坐标系相同

    # 将截图写入视频文件
    # video_writer.write(image)

    if time.time() - start > 75:
        running = False
        for drone in drones:
            Coor_x.append(drone.history_x)
            Coor_y.append(drone.history_y)


# video_writer.release()
pygame.quit()

from matplotlib import pyplot as plt

plt.plot(Time, Aggregation, '-',label='Aggregation')
plt.plot(Time, Alignment, '--', label='Alignment')
plt.plot(Time, Divergence, '-.', label='Divergence')
plt.ylim(-0.05, 1.05)
plt.xlim(xmin=0)
plt.xlabel('Second')
plt.ylabel('Probability')
plt.legend(loc='upper right')
plt.title('Graph of probability changes in ICN judgment of different intentions')
plt.savefig('Prob-AGG.png', dpi=300)
plt.show()

marker = ['.', '+', 'x', '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', 'D', 'd', '8', '_', 'o', 'v', '^', '<', '>']
step_diff = len(Coor_x[0]) - len(Prediction)
line_true, line_pre = None, None
Pre_x, Pre_y = [], []
for i in range(20):
    Pre_x.append([Prediction[j][i*2] for j in range(0, len(Prediction), 40)])
    Pre_y.append([Prediction[j][i*2+1] for j in range(0, len(Prediction), 40)])
for i in range(20):
    for j in range(0, len(Coor_x[0]), 40):
        line_true, = plt.plot(Coor_x[i][j], Coor_y[i][j], marker[i]+'-b', ms=5)
for i in range(20):
    line_pre, = plt.plot(Pre_x[i], Pre_y[i], marker[i]+'-r', ms=5)
plt.legend(handles=[line_true, line_pre], labels=['True path', 'Predicted path'])
plt.ylim(ymin=500, ymax=0)
plt.xlabel('x')
plt.ylabel('y')
plt.title('The true and predicted path of the UAV swarm')
plt.savefig('Path-AGG.png', dpi=300)
plt.show()

point = None
for i in range(20):
    for j in range(0, len(Time), 5):
        error = np.sqrt((Coor_x[i][j+5] - Prediction[j][i*2])**2 + (Coor_y[i][j+5] - Prediction[j][i*2+1])**2)
        point, = plt.plot(Time[j], error, '.c')
plt.xlabel('Second')
plt.ylabel('Meter')
plt.title('Scatter of the error of the prediction of the path')
# plt.ylim(ymax=7.05)
plt.legend(handles=[point], labels=['Error'], loc='upper right')
plt.savefig('Error-AGG.png', dpi=300)
plt.show()


