"""

Mobile robot motion planning sample with Dynamic Window Approach

author: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı

"""

import math
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

show_animation = True


def dwa_control(x, config, goal, ob):
    """
    Dynamic Window Approach control
    x: 机器人状态[x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    config: 配置参数
    goal: 目标点
    ob: 障碍物
    """

    # 计算速度窗口
    dw = calc_dynamic_window(x, config)

    # 计算控制指令，获取轨迹
    u, trajectory = calc_control_and_trajectory(x, dw, config, goal, ob)

    return u, trajectory


class RobotType(Enum):
    circle = 0
    rectangle = 1


class Config:
    """
    simulation parameter class
    """

    def __init__(self):

        # 最大速度，最小速度 [m/s]
        self.max_speed = 1.0
        self.min_speed = -0.5

        # 最大角速度 [rad/s]
        self.max_yaw_rate = 40.0 * math.pi / 180.0

        # 最大加速度 [m/ss]
        self.max_accel = 0.2
        # 最大角加速度 [rad/ss]
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  

        # 速度 [m/s] 和角速度 [rad/s] 的采样分辨率
        self.v_resolution = 0.01
        self.yaw_rate_resolution = 0.1 * math.pi / 180.0

        # 时间步长 [s] 
        self.dt = 0.1
        # 轨迹预测的时间长度 [s] 
        self.predict_time = 3.0

        # 代价函数系数
        self.to_goal_cost_gain = 0.15
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 1.0

        # constant to prevent robot stucked
        self.robot_stuck_flag_cons = 0.001  

        # circle = 0;   rectangle = 1
        self.robot_type = RobotType.circle

        # 机器人几何尺寸
        self.robot_radius = 1.0  # [m] for collision check
        self.robot_width = 0.5   # [m] for collision check
        self.robot_length = 1.2  # [m] for collision check


        # 障碍物的坐标 [x(m) y(m), ....]
        self.ob = np.array([[-1, -1],
                            [0, 2],
                            [4.0, 2.0],
                            [5.0, 4.0],
                            [5.0, 5.0],
                            [5.0, 6.0],
                            [5.0, 9.0],
                            [8.0, 9.0],
                            [7.0, 9.0],
                            [8.0, 10.0],
                            [9.0, 11.0],
                            [12.0, 13.0],
                            [12.0, 12.0],
                            [15.0, 15.0],
                            [13.0, 13.0]
                            ])

    @property
    def robot_type(self):
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value):
        if not isinstance(value, RobotType):
            raise TypeError("robot_type must be an instance of RobotType")
        self._robot_type = value


config = Config()


def motion(x, u, dt):
    """
    motion model
    """

    # x，y，yaw
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[2] += u[1] * dt
    
    # v，w
    x[3] = u[0]
    x[4] = u[1]

    return x


def calc_dynamic_window(x, config):
    """
    calculation dynamic window based on current state x
    """

    # 机器人的速度空间，速度边界限制
    Vs = [config.min_speed, config.max_speed,
          -config.max_yaw_rate, config.max_yaw_rate]

    # 一个时间步内机器人可达到的速度，加速度限制
    # x[3]和x[4]是机器人当前速度
    # [v - a_v * dt, v + a_v * dt]; [w - a_w * dt, w + a_w * dt]
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_delta_yaw_rate * config.dt,
          x[4] + config.max_delta_yaw_rate * config.dt]

    #  Vs和Vd的交集     [v_min, v_max, yaw_rate_min, yaw_rate_max]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]
    
    # 理论上还有一个障碍物限制，但该条件在采样初期无法得到，需要先使用Vs和Vd的交集的速度组合采样模拟出轨迹后
    # 计算当前速度下对应模拟轨迹与障碍物之间的最近距离，然后看当前采样的这对速度能否在碰到障碍物之前停下 
    # 如果可以，则这对速度是可接受的，否则需要抛弃掉

    return dw


def predict_trajectory(x_init, v, y, config):
    """
    预测机器人在给定控制输入下的轨迹
    """
    # 机器人当前状态
    x = np.array(x_init)

    # 轨迹
    trajectory = np.array(x)

    # 预测时刻
    time = 0

    # 预测时间内进行轨迹推算，config.predict_time=3
    while time <= config.predict_time:

        # 计算给定这组速度的推算轨迹状态
        x = motion(x, [v, y], config.dt)

        # np.vstack 用于将两个数组在垂直方向（按行）堆叠起来
        trajectory = np.vstack((trajectory, x))

        time += config.dt

    return trajectory


def calc_control_and_trajectory(x, dw, config, goal, ob):
    """
    calculation final input with dynamic window
    x: 机器人状态[x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    dw: 搜索空间
    config: 配置参数
    goal: 目标点
    ob: 障碍物
    """

    # 初始状态（机器人的当前状态）
    x_init = x[:]

    # 当前最优轨迹的代价，初始值为无穷大
    min_cost = float("inf")

    # 最优控制指令，初始值为 [0.0, 0.0]，表示线速度和角速度
    best_u = [0.0, 0.0]

    # 最优轨迹，初始值为机器人的当前状态
    best_trajectory = np.array([x])

    # 在动态窗口内遍历所有可能的线速度 v 和角速度 y，并且计算每组速度的代价，挑选出最优控制指令
    # 线速度：dw[0]到dw[1]之间，按照config.v_resolution=0.01进行采样
    for v in np.arange(dw[0], dw[1], config.v_resolution):

        # 角速度：dw[2]到dw[3]之间，按照config.yaw_rate_resolution=0.1 * math.pi / 180.0进行采样
        for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):

            # 对搜索空间中采样的每组速度都进行轨迹推算
            trajectory = predict_trajectory(x_init, v, y, config)

            # 计算每组速度推算轨迹的代价

            # 评估机器人在轨迹末端的朝向是否与目标点方向一致
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)

            # 推算轨迹末端点线速度越大, 代价越小；线速度越小，代价越大
            speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])

            # 1/min_r，离障碍物越近, 代价越大; 离障碍物越远, 代价越小
            ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(trajectory, ob, config)

            final_cost = to_goal_cost + speed_cost + ob_cost

            # min_cost是当前最优代价，初始时为无穷大
            # 乳沟当前轨迹的代价小于等于最小代价
            if min_cost >= final_cost:
                
                # 当计算得到的最新代价小于等于min_cost的是时候，用最新代价更新min_cost
                min_cost = final_cost

                # final_cost对应的速度向量记录为最优控制量
                best_u = [v, y]
                # final_cost对应的轨迹记录为最优轨迹
                best_trajectory = trajectory

                # 如果机器人可能卡住（线速度和角速度都接近于零），则调整角速度以避免卡住
                # config.robot_stuck_flag_cons = 0.001
                if abs(best_u[0]) < config.robot_stuck_flag_cons \
                        and abs(x[3]) < config.robot_stuck_flag_cons:
                    # to ensure the robot do not get stuck in
                    # best v=0 m/s (in front of an obstacle) and
                    # best omega=0 rad/s (heading to the goal with
                    # angle difference of 0)
                    best_u[1] = -config.max_delta_yaw_rate
            
    return best_u, best_trajectory


def calc_obstacle_cost(trajectory, ob, config):
    """
    calc obstacle cost inf: collision
    如果轨迹与障碍物发生碰撞, 则返回无穷大(Inf), 否则返回一个与轨迹到最近障碍物距离成反比的代价
    """

    # 提取所有障碍物的 x 和 y 坐标
    ox = ob[:, 0]
    oy = ob[:, 1]

    # 计算轨迹中每个点的 x和y 坐标与所有障碍物的 x和y 坐标的差值
    dx = trajectory[:, 0] - ox[:, None]
    dy = trajectory[:, 1] - oy[:, None]

    # 计算轨迹中每个点到所有障碍物的欧几里得距离，结果是一个二维数组，形状为 (轨迹点数, 障碍物数)
    r = np.hypot(dx, dy)

    # 检查机器人的边界是否与障碍物发生碰撞
    # 矩形
    if config.robot_type == RobotType.rectangle:

        # 提取轨迹中每个点的朝向角 
        yaw = trajectory[:, 2]

        # 计算每个点的旋转矩阵，用于将障碍物坐标转换到机器人的局部坐标系
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        rot = np.transpose(rot, [2, 0, 1])

        # 将障碍物坐标转换到机器人的局部坐标系，方便检查是否在机器人的边界内
        local_ob = ob[:, None] - trajectory[:, 0:2]
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        local_ob = np.array([local_ob @ x for x in rot])
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])

        # 边界检查
        upper_check = local_ob[:, 0] <= config.robot_length / 2
        right_check = local_ob[:, 1] <= config.robot_width / 2
        bottom_check = local_ob[:, 0] >= -config.robot_length / 2
        left_check = local_ob[:, 1] >= -config.robot_width / 2

        # 如果任何一个障碍物在机器人的边界内，则返回无穷大（Inf），表示发生碰撞
        if (np.logical_and(np.logical_and(upper_check, right_check),
                           np.logical_and(bottom_check, left_check))).any():
            return float("Inf")
    
    # 圆形
    elif config.robot_type == RobotType.circle:

        if np.array(r <= config.robot_radius).any():
            return float("Inf")
        
    # 找到轨迹点到所有障碍物的最小距离
    min_r = np.min(r)

    # 返回避障代价
    # 离障碍物越近, 代价越大; 离障碍物越远, 代价越小
    return 1.0 / min_r  # OK


def calc_to_goal_cost(trajectory, goal):
    """
        轨迹末端与目标点的角度偏差
    """

    # 轨迹末端点到目标点的位置偏差
    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]

    # 计算目标点与轨迹末端点连线的角度（以弧度表示）
    error_angle = math.atan2(dy, dx)

    # 计算目标点与轨迹末端连线的角度 error_angle 与轨迹末端点的朝向角 trajectory[-1, 2] 之间的差值
    # 差值表示机器人当前朝向与目标点方向之间的偏差
    cost_angle = error_angle - trajectory[-1, 2]

    # 计算 cost_angle 的正弦和余弦值
    # 使用 math.atan2 将角度偏差规范化到 [−π,π] 范围内
    # 取规范化后的角度偏差的绝对值，确保代价始终为非负数
    cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

    return cost


def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    
    plt.plot(x, y)


def plot_robot(x, y, yaw, config):  # pragma: no cover

    if config.robot_type == RobotType.rectangle:
        outline = np.array([[-config.robot_length / 2, config.robot_length / 2,
                             (config.robot_length / 2), -config.robot_length / 2,
                             -config.robot_length / 2],
                            [config.robot_width / 2, config.robot_width / 2,
                             - config.robot_width / 2, -config.robot_width / 2,
                             config.robot_width / 2]])
        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                         [-math.sin(yaw), math.cos(yaw)]])
        outline = (outline.T.dot(Rot1)).T
        outline[0, :] += x
        outline[1, :] += y
        plt.plot(np.array(outline[0, :]).flatten(),
                 np.array(outline[1, :]).flatten(), "-k")
        
    elif config.robot_type == RobotType.circle:
        circle = plt.Circle((x, y), config.robot_radius, color="b")
        plt.gcf().gca().add_artist(circle)
        out_x, out_y = (np.array([x, y]) +
                        np.array([np.cos(yaw), np.sin(yaw)]) * config.robot_radius)
        plt.plot([x, out_x], [y, out_y], "-k")


# 主程序
def main(gx=10.0, gy=10.0, robot_type=RobotType.circle):
    print(__file__ + " start!!")

    # 机器人初始状态 [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    x = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    # 目标点位置，默认(10, 10)， [x(m), y(m)]
    goal = np.array([gx, gy])

    # 机器人类型，默认为圆
    config.robot_type = robot_type

    # 定义并初始化轨迹变量
    trajectory = np.array(x)

    # 加载障碍物位置
    ob = config.ob

    # 主循环
    while True:

        # 调用 dwa_control 函数计算最优控制输入和预测轨迹
        u, predicted_trajectory = dwa_control(x, config, goal, ob)

        # 使用 motion 函数更新机器人的状态
        x = motion(x, u, config.dt)

        # 将状态存储到轨迹中
        trajectory = np.vstack((trajectory, x))

        # 动态显示
        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")
            plt.plot(x[0], x[1], "xr")
            plt.plot(goal[0], goal[1], "xb")
            plt.plot(ob[:, 0], ob[:, 1], "ok")
            plot_robot(x[0], x[1], x[2], config)
            plot_arrow(x[0], x[1], x[2])
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)

        # 检查是否到达目标点
        # 计算机器人位置和目标点的距离
        dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
        # 距离小于机器人半径则表示到达目标点，退出循环
        if dist_to_goal <= config.robot_radius:
            print("Goal!!")
            break
    
    # 程序结束时绘制最终轨迹并显示
    print("Done")
    if show_animation:
        plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")
        plt.pause(0.0001)
        plt.show()


if __name__ == '__main__':
    main(robot_type=RobotType.rectangle)
    # main(robot_type=RobotType.circle)