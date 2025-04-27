import numpy as np
import math
import random

import matplotlib.pyplot as plt


def update_robot_state(state, cmd, dt):
    state[0] += cmd[0] * math.cos(state[2]) * dt
    state[1] += cmd[0] * math.sin(state[2]) * dt
    state[2] += cmd[1] * dt

    state[3] = cmd[0]
    state[4] = cmd[1]

    return state


class DWA():
    def __init__(self):
        # 采样周期
        self.dt = 0.1
        # 预测时域
        self.predict_time = 3

        # 机器人半径
        self.robot_radius = 1.0
        # 障碍物半径
        self.obstacle_radius = 0.2

        # 线速度约束 m/s
        self.v_min = -0.5
        self.v_max = 1.0
        # 角速度约束 rad/s
        self.w_min = -40.0 * math.pi / 180.0
        self.w_max = 40.0 * math.pi / 180.0

        # 加速度约束 m/s^2
        self.v_acc_max = 0.2
        # 角加速度约束 rad/s^2
        self.w_acc_max = 40.0 * math.pi / 180.0

        # 速度采样分辨率
        self.v_resolution = 0.02
        self.w_resolution = 0.02

        # 评价函数权重
        self.weight_heading  = 0.15
        self.weight_obstacle = 0.4
        self.weight_velocity = 1.0

        # 避免机器人卡死
        self.robot_stuck_flag = 0.001
        

    def dwa_control(self, robot_state, goal, obstacles):

        # 1. 生成一个当前速度空间的动态窗口
        allowable_v, allowable_w = self._get_dynamic_windows(robot_state)

        # 2. 在动态窗口中进行速度采样, 推算出预测时域内的所有候选轨迹(返回的是列表，列表中每个元素是np.ndarray)
        candidate_trajectory = self._generate_trajectory(robot_state, allowable_v, allowable_w)

        # 3. 评价所有候选轨迹, 选择最优轨迹
        opt_vel, opt_trajectory = self._evaluate_trajectory(candidate_trajectory, robot_state, goal, obstacles)

        return opt_vel, opt_trajectory


    def _get_dynamic_windows(self, robot_state):
        '''
        根据机器人的当前状态robot_state(x,y,yaw,v,w), 以及速度和加速度约束，生成一个速度空间的动态窗口
        '''
        # 速度边界限制
        Vs = [self.v_min, self.v_max, self.w_min, self.w_max]

        # 加速度限制, robot_state[3]和robot_state[4]是机器人当前速度(v, w)
        Vd = [robot_state[3] - self.v_acc_max * self.dt,
              robot_state[3] + self.v_acc_max * self.dt,
              robot_state[4] - self.w_acc_max * self.dt,
              robot_state[4] + self.w_acc_max * self.dt ]

        # 动态窗口
        allowable_v = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1])]
        allowable_w = [max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

        return allowable_v, allowable_w


    def _generate_trajectory(self, robot_state, allowable_v, allowable_w):
        '''
        在速度空间的动态窗口中进行速度采样(v_sample, w_sample)
        根据机器人当前状态robot_state(x,y,yaw,v,w), 推算每组采样速度在预测时域内的轨迹
        所有采样速度生成的轨迹保存在candidate_trajectory中, 作为当前机器人的候选轨迹
        '''
        candidate_trajectory = []

        # 在速度空间中按照预先设定的分辨率采样
        v_num = int((allowable_v[1] - allowable_v[0]) / self.v_resolution) + 1
        w_num = int((allowable_w[1] - allowable_w[0]) / self.w_resolution) + 1

        for v_sample in np.linspace(allowable_v[0], allowable_v[1], v_num):
            for w_sample in np.linspace(allowable_w[0], allowable_w[1], w_num):
                # 推算一组采样速度(v_sample, w_sample)在预测时域内的轨迹(返回类型是np.ndarray)
                sample_trajectory = self._trajectory_predict(robot_state, v_sample, w_sample)

                # 将每组采样速度生成的轨迹添加到候选轨迹中
                candidate_trajectory.append(sample_trajectory)

        return candidate_trajectory
    

    def _evaluate_trajectory(self, candidate_trajectory, robot_state, goal, obstacles):
        '''
        评价所有候选轨迹, 选择最优轨迹
        '''
        # todo:
        # 处理障碍物返回无穷的情况，可直接去掉；
        # 处理所有预测轨迹都碰撞的情况；
        # 重写归一化 
        # 核查清楚代码中变量维数的问题 
        # 整理出一份文档
        # 添加到目标的直线距离的代价函数
        # 移植到ir-sim
        # 
        
        # 计算每条候选轨迹的代价值, 去掉碰撞轨迹 (candidate_trajectory是列表, 其中每个元素都是np.ndarray, (x,y,yaw,v,w))
        heading_cost  = []
        obstacle_cost = []
        velocity_cost = []

        for trajectory in candidate_trajectory:
            
            # 去掉碰撞轨迹
            obs_cost = self._obstacle_cost(trajectory, obstacles)
            if obs_cost == float('inf'):
                continue

            heading_cost.append(self._heading_cost(trajectory, goal))
            obstacle_cost.append(obs_cost)
            velocity_cost.append(self._velocity_cost(trajectory))

        # 如果所有轨迹都碰撞的情况, 则生成保持原地的虚拟轨迹, 让机器人原地旋转脱困
        # if not obstacle_cost:
        #     stationary_velocity = [0.0, self.w_max * 0.5 * (1 if random.random() > 0.5 else -1)]
        #     stationary_trajectory = np.repeat([np.append(robot_state[:3], stationary_velocity)], 
        #                                     int(self.predict_time/self.dt) + 1, 
        #                                     axis=0)
        #     return stationary_velocity, stationary_trajectory
        
        # 归一化处理(加强数值稳定性, 避免除以零))
        heading_cost  = self._normalize_cost(heading_cost)
        obstacle_cost = self._normalize_cost(obstacle_cost)
        velocity_cost = self._normalize_cost(velocity_cost)


        # 定义最优代价, 初始化为无穷大
        min_cost = float('inf')
        # 定义最优控制量, 初始化为0
        opt_vel = [0.0, 0.0]
        # 定义最优轨迹, 初始化为当前状态
        opt_trajectory = np.array(robot_state)
        # 评估有效轨迹(去掉了碰撞轨迹)
        for i in range( len(obstacle_cost) ):
            # 第i条有效轨迹的代价值
            total_cost_i = ( self.weight_heading  * heading_cost[i]  +
                             self.weight_obstacle * obstacle_cost[i] +
                             self.weight_velocity * velocity_cost[i] )
            # 保留代价值最小的轨迹，作为最优输出
            if total_cost_i < min_cost:
                min_cost = total_cost_i
                opt_trajectory = candidate_trajectory[i]
                opt_vel = [opt_trajectory[-1,3], opt_trajectory[-1,4]]

                # 避免机器人卡死
                if abs(opt_vel[0]) < self.robot_stuck_flag and abs(robot_state[3]) < self.robot_stuck_flag:
                    opt_vel[1] = -self.w_acc_max

        return opt_vel, opt_trajectory


    def _heading_cost(self, trajectory, goal):
        '''
        评价在当前采样速度下产生的轨迹终点位置方向与目标点连线夹角的误差
        '''
        # 轨迹末端点到目标点的位置偏差
        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        # 目标点与轨迹末端点连线的角度
        angle_h2r = math.atan2(dy, dx)

        # 目标点与轨迹末端连线的角度 angle_h2r 与轨迹末端点的朝向角 trajectory[-1, 2] 的差值作为代价
        # 归一化到[0, π], 避免 ±π 跳变, 并线性表示角度偏差大小 (适合作为代价函数，值越小表示对准越好)
        return abs((angle_h2r - trajectory[-1][2] + math.pi) % (2 * math.pi) - math.pi)
    

    def _obstacle_cost(self, trajectory, obstacles):
        '''
        评价在当前采样速度下产生的轨迹与障碍物之间的最近距离
        '''
        # 提取所有障碍物的 x 和 y 坐标
        ob_x = obstacles[:, 0]
        ob_y = obstacles[:, 1]

        # 计算轨迹中每个点的 x和y 坐标与所有障碍物的 x和y 坐标的差值
        dx = trajectory[:, 0] - ob_x[:, None]
        dy = trajectory[:, 1] - ob_y[:, None]

        # 计算轨迹中每个点到所有障碍物的欧几里得距离
        r = np.hypot(dx, dy)

        # 碰撞检测
        # 需要根据机器人的几何形状范围进行判断, 此处直接处理成圆形区域, 区域半径为robot_radius+obstacle_radius
        if np.array(r <= self.robot_radius + self.obstacle_radius).any():
            return float("Inf")

        return 1 / np.min(r)


    def _velocity_cost(self, trajectory):
        '''
        评价在当前采样速度下产生的轨迹的线速度大小
        '''
        return self.v_max - trajectory[-1, 3]


    def _trajectory_predict(self, robot_state, v_sample, w_sample):
        '''
        根据机器人当前状态robot_state(x,y,yaw,v,w), 推算一组采样速度(v_sample, w_sample)在预测时域内的轨迹
        '''
        # 定义并初始化机器人的预测状态, 初始状态为机器人当前状态
        predict_state = np.array(robot_state)
        # 定义并初始化机器人的轨迹(存储), 初始轨迹点为机器人当前状态
        predict_trajectory = predict_state

        # 在预测时域内, 计算(v_sample, w_sample)这组速度生成的轨迹
        time = 0
        while time <= self.predict_time:
            predict_state = self._diff_kinematic_model(predict_state, [v_sample, w_sample], self.dt)
            predict_trajectory = np.vstack((predict_trajectory, predict_state))
            time += self.dt
        
        return predict_trajectory
    

    def _normalize_cost(self, cost):
        '''
        归一化处理
        '''
        cost = np.array(cost)
        min_val, max_val = np.min(cost), np.max(cost)
        if max_val == min_val:
            return np.zeros_like(cost)
        return (cost - min_val) / (max_val - min_val)


    def _diff_kinematic_model(self, state, cmd, dt):
        '''
        差速移动机器人运动学模型
        '''
        state[0] += cmd[0] * math.cos(state[2]) * dt
        state[1] += cmd[0] * math.sin(state[2]) * dt
        state[2] += cmd[1] * dt

        state[3] = cmd[0]
        state[4] = cmd[1]

        return state


def plot_arrow(x, y, yaw, length = 0.5, width = 0.1):
    plt.arrow(x, 
              y, 
              length * math.cos(yaw), 
              length * math.sin(yaw),
              head_length = width, 
              head_width = width
              )
    plt.plot(x, y)

def plot_robot(x, y, yaw, dwa):
    circle = plt.Circle((x, y), dwa.robot_radius, color="b")
    plt.gcf().gca().add_artist(circle)
    out_x, out_y = (np.array([x, y]) +
                    np.array([np.cos(yaw), np.sin(yaw)]) * dwa.robot_radius)
    plt.plot([x, out_x], [y, out_y], "-k")


if __name__ == '__main__':

    print(__file__ + " start!!")

    # 机器人初始状态 [x(m), y(m), yaw(rad), v(m/s), w(rad/s)]
    # robot_state = np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0])
    robot_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    # 目标点位置 [x(m), y(m)]
    goal = np.array([10.0, 0.0])

    # 障碍物位置
    obstacles = np.array([[-1, -1],
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

    # 定义并初始化机器人的轨迹(存储)
    robot_trajectory = np.array(robot_state)

    # 实例化DWA
    dwa = DWA()

    step = 0
    while True:
        # 调用DWA, 得到机器人控制指令和最优轨迹
        opt_vel, opt_trajectory = dwa.dwa_control(robot_state, goal, obstacles)
        
        # 更新机器人状态(x,y,yaw,v,w)
        robot_state = update_robot_state(robot_state, opt_vel, dwa.dt)

        print('sim step:', step)
        print('best cmd: ', opt_vel)
        print('best trajectory: ', opt_trajectory)
        print('robot state: ', robot_state)
        print('----------------------------------------------------------------------------')

        # 记录轨迹
        robot_trajectory = np.vstack((robot_trajectory, robot_state))

        # 按 esc 键退出
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        
        # 可视化
        plt.cla()
        # 绿色实现绘制预测轨迹
        plt.plot(opt_trajectory[:, 0], opt_trajectory[:, 1], "-g")
        # 红色叉号标记机器人当前位置
        plt.plot(robot_state[0], robot_state[1], "xr")
        # 蓝色叉号标记目标位置
        plt.plot(goal[0], goal[1], "xb")
        # 黑色圆圈标记障碍物位置
        plt.plot(obstacles[:, 0], obstacles[:, 1], "ok")
        # 绘制机器人的形状
        plot_robot(robot_state[0], robot_state[1], robot_state[2], dwa)
        # 绘制机器人方向的箭头
        plot_arrow(robot_state[0], robot_state[1], robot_state[2])
        # 设置 x 轴和 y 轴的比例相同
        plt.axis("equal")
        # 显示网格线
        plt.grid(True)
        # 程序暂停 0.001 秒（即 1 毫秒），然后继续执行后续代码 (控制图形的刷新频率)
        plt.pause(0.001)

        step += 1

        # 判断机器人是否到达目标点
        dist_to_goal = math.hypot(robot_state[0] - goal[0], robot_state[1] - goal[1])
        if dist_to_goal <= dwa.robot_radius:
            print("Goal!!!")
            break
    
    # 仿真结束
    print("Done")
    # 红色实线绘制机器人轨迹
    plt.plot(robot_trajectory[:, 0], robot_trajectory[:, 1], "-r")
    plt.pause(0.001)
    # 显示当前绘制的图形
    plt.show()
