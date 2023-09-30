"""
__title__    = BoidsAR.py
__author__   = 'Hansen Wong // 王之超'
__time__     = 2023/3/6 15:37
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

from New_AR_func import *
import datetime

A_sigma = -5

for test in range(5):
    drones = run(A_sigma)


    # 数据存储
    data = pd.DataFrame({})
    i = 0
    for drone in drones:
        data.insert(i, 'Drone ' + str(int(i / 4) + 1) + ' x', drone.history_x)
        data.insert(i + 1, 'Drone ' + str(int(i / 4) + 1) + ' y', drone.history_y)
        data.insert(i + 2, 'Drone ' + str(int(i / 4) + 1) + ' vx', drone.history_vx)
        data.insert(i + 3, 'Drone ' + str(int(i / 4) + 1) + ' vy', drone.history_vy)
        i += 4

    print(data)

    intention = np.zeros_like(drones[0].history_x)
    if A_sigma > 0:
        intention += 2
    elif A_sigma == 0:
        intention += 1
    else:
        pass
    data.insert(80, 'Intention', intention)

    # dt = str(datetime.date.today()) + '-' + str(datetime.datetime.now().hour) + '-' + str(datetime.datetime.now().minute) + '-' + str(datetime.datetime.now().second)
    # data.to_csv('dataset_new/Divergence/' + dt + '.csv')
    # data.to_csv('dataset_new/Sum/' + dt + '.csv')
