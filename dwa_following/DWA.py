import numpy as np
import math

from animation import Animation_robot


class Two_wheeled_robot():
    '''
    两轮差动机器人模型
    '''
    def __init__(self, init_x, init_y, init_theta):
        # 初始化机器人状态(x, y, theta, v, omega)
        self.x = init_x
        self.y = init_y
        self.th = init_theta
        self.u_v = 0.0
        self.u_th = 0.0

        # 存储机器人的轨迹(x, y, theta, v, omega)
        self.traj_x = [init_x]
        self.traj_y = [init_y]
        self.traj_th = [init_theta]
        self.traj_u_v = [0.0]
        self.traj_u_th = [0.0]

    def update_state(self, u_v, u_th, dt):
        '''
        根据控制量(u_v, u_th)更新机器人状态
        '''
        # 更新控制量
        self.u_v = u_v
        self.u_th = u_th

        # 差速机器人运动学模型
        # self.x += self.u_v * math.cos(self.th) * dt
        # self.y += self.u_v * math.sin(self.th) * dt
        # self.th += self.u_th * dt
        # 根据速度和角速度计算下一状态
        next_x = self.x + self.u_v * math.cos(self.th) * dt
        next_y = self.y + self.u_v * math.sin(self.th) * dt
        next_th = self.th + self.u_th * dt

        # 存储机器人轨迹
        # self.traj_x.append(self.x)
        # self.traj_y.append(self.y)
        # self.traj_th.append(self.th)
        self.traj_x.append(next_x)
        self.traj_y.append(next_y)
        self.traj_th.append(next_th)

        # 更新状态
        self.x = next_x
        self.y = next_y
        self.th = next_th

        return self.x, self.y, self.th


class Const_goal():
    def __init__(self):
        self.traj_g_x = []
        self.traj_g_y = []

    def calc_goal(self, time_step):
        '''
        根据时间步生成目标点
        '''
        if time_step <= 100:
            g_x = 10.0
            g_y = 10.0
        else:
            g_x = -10.0
            g_y = -10.0

        self.traj_g_x.append(g_x)
        self.traj_g_y.append(g_y)

        return g_x, g_y


class Obstacle():
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size


class Simulator_DWA_robot():
    '''
    根据机器人当前位置，以及给定的控制量，在预测周期内推算机器人轨迹
    '''
    def __init__(self):

        # 最大加速度 m/s^2
        self.max_accelation = 1.0
        # 最大角加速度 rad/s^2
        self.max_ang_accelation = 100 * math.pi / 180

        # 线速度最大值和最小值 m/s
        self.lim_max_velo = 1.6
        self.lim_min_velo = 0.0

        # 角速度最大值和最小值 rad/s
        self.lim_max_ang_velo = math.pi
        self.lim_min_ang_velo = -math.pi

    def predict_state(self, ang_velo, velo, x, y, th, dt, pre_step):
        next_xs = []
        next_ys = []
        next_ths = []

        # 预测时间pre_time=3, pre_step=30, dt=0.1
        for i in range(pre_step):
            # 差速机器人运动学模型，当前控制量和当前位置，推算预测周期内的轨迹
            temp_x = x + velo * math.cos(th) * dt
            temp_y = y + velo * math.sin(th) * dt
            temp_th = th + ang_velo * dt

            # 存储预测周期内的推算轨迹
            next_xs.append(temp_x)
            next_ys.append(temp_y)
            next_ths.append(temp_th)

            x = temp_x
            y = temp_y
            th = temp_th

        return next_xs, next_ys, next_ths


class Path():
    # 封装一段轨迹的状态和控制量
    def __init__(self, u_th, u_v): 
        self.x = None
        self.y = None
        self.th = None
        self.u_v = u_v  # 线速度
        self.u_th = u_th  # 角速度


class DWA():
    '''
    DWA控制器
    '''
    def __init__(self):
        self.simu_robot = Simulator_DWA_robot()

        self.pre_time = 3  # 预测时间（秒）
        self.pre_step = 30  # 预测步数

        # 速度采样分辨率
        self.delta_velo = 0.02  # 速度增量
        self.delta_ang_velo = 0.02  # 角速度增量

        self.samplingtime = 0.1  # 采样时间

        self.weight_angle = 0.04  # 角度权重
        self.weight_velo = 0.2  # 速度权重
        self.weight_obs = 0.1  # 障碍物权重

        self.traj_paths = []  # 存储所有路径
        self.traj_opt = []  # 存储最优路径

    def calc_input(self, g_x, g_y, state, obstacles):
        # 所有采样速度生成的所有预测轨迹
        paths = self._make_path(state)

        # 评估所有路径，选择最优路径
        opt_path = self._eval_path(paths, g_x, g_y, state, obstacles)

        # 保存最优路径
        self.traj_opt.append(opt_path)

        return paths, opt_path


    def _make_path(self, state):
        
        # 存储所有采样速度生成的所有预测轨迹
        paths = []

        # 1. 构建速度空间窗口，计算角速度和线速度的范围
        min_ang_velo, max_ang_velo, min_velo, max_velo = self._calc_range_velos(state)

        # 2. 速度采样，在速度空间中按照预先设定的分辨率采样(self.delta_velo, self.delta_ang_velo)
        for ang_velo in np.arange(min_ang_velo, max_ang_velo, self.delta_ang_velo):
            for velo in np.arange(min_velo, max_velo, self.delta_velo):
                # 当前一组采样速度
                path = Path(ang_velo, velo)

                # next_x, next_y, next_th为列表，存储预测周期内的轨迹
                next_x, next_y, next_th = self.simu_robot.predict_state(ang_velo, velo, \
                                                                        state.x, state.y, state.th, \
                                                                        self.samplingtime, self.pre_step)
                
                path.x = next_x
                path.y = next_y
                path.th = next_th

                # 将当前一组采样速度生成的路径加入到所有路径中
                paths.append(path)

        # 保存所有路径
        self.traj_paths.append(paths)  

        return paths
    

    # 相当于_get_dynamic_windows
    def _calc_range_velos(self, state):
        # 计算角速度可变化范围：采样周期*最大角加速度 
        range_ang_velo = self.samplingtime * self.simu_robot.max_ang_accelation

        # ##### 有问题 ！！！！ 角速度是有上限，如self.lim_max_ang_velo = math.pi
        # 用当前角速度加减角速度可变化范围，可能会超过限制
        # ！！！！
        # 角速度最小值：当前角速度-角速度可变化范围
        min_ang_velo = state.u_th - range_ang_velo
        # 角速度最大值：当前角速度+角速度可变化范围
        max_ang_velo = state.u_th + range_ang_velo

        # 对角速度边界值进行限制
        if min_ang_velo < self.simu_robot.lim_min_ang_velo:
            min_ang_velo = self.simu_robot.lim_min_ang_velo
        
        if max_ang_velo > self.simu_robot.lim_max_ang_velo:
            max_ang_velo = self.simu_robot.lim_max_ang_velo
        
        # 计算线速度可变化范围：采样周期*最大加速度 
        range_velo = self.samplingtime * self.simu_robot.max_accelation

        min_velo = state.u_v - range_velo
        max_velo = state.u_v + range_velo

        if min_velo < self.simu_robot.lim_min_velo:
            min_velo = self.simu_robot.lim_min_velo

        if max_velo > self.simu_robot.lim_max_velo:
            max_velo = self.simu_robot.lim_max_velo

        return min_ang_velo, max_ang_velo, min_velo, max_velo
    

    def _eval_path(self, paths, g_x, g_y, state, obstacles):

        # 计算离机器人当前位置最近的障碍物
        nearest_obs = self._calc_nearest_obs(state, obstacles)


    def _calc_nearest_obs(self, state, obstacles):
        # 定义最近的障碍物列表
        nearest_obs = []

        # 机器人到障碍物距离的阈值
        area_dis_to_obs = 5

        # 遍历所有障碍物
        for obs in obstacles:

            # 机器人当前位置到障碍物的距离
            temp_dis_to_obs = math.sqrt((state.x - obs.x) ** 2 + (state.y - obs.y) ** 2)

            # 如果机器人到障碍物的距离小于area_dis_to_obs，则将该障碍物加入到最近的障碍物列表中
            if temp_dis_to_obs < area_dis_to_obs:
                # 将该障碍物加入到最近的障碍物列表中
                nearest_obs.append(obs)

        return nearest_obs



class Main_controller():
    def __init__(self):

        # 初始化采样时间
        self.samplingtime = 0.1

        # 初始化机器人，初始位置(x, y, theta)=(0, 0, 0)
        self.robot = Two_wheeled_robot(0.0, 0.0, 0.0)

        # 初始化目标
        self.goal_maker = Const_goal()

        # 初始化障碍物(x, y, size)
        self.obstacles = [Obstacle(4, 1, 0.25), Obstacle(0, 4.5, 0.25), 
                          Obstacle(3, 4.5, 0.25), Obstacle(5, 3.5, 0.25), 
                          Obstacle(7.5, 9.0, 0.25)]
        
        # 初始化DWA控制器
        self.controller = DWA()

    def run_to_goal(self):
        goal_flag = False
        time_step = 0

        while not goal_flag:
            # 当前目标位置(x, y)
            g_x, g_y = self.goal_maker.calc_goal(time_step)

            # 调用DWA控制器，得到机器人控制指令opt_path和最优轨迹
            paths, opt_path = self.controller.calc_input(g_x, g_y, self.robot, self.obstacles)

            # dwa输出的最优控制指令
            u_th = opt_path.u_th
            u_v = opt_path.u_v

            # 根据控制指令更新机器人状态
            self.robot.update_state(u_th, u_v, self.samplingtime)

            # 计算机器人与目标点的距离
            dis_to_goal = np.sqrt((g_x - self.robot.x)**2 + (g_y - self.robot.y)**2)
            # 人机相对距离小于0.5米，即认为到达目标
            if dis_to_goal < 0.5:
                goal_flag = True

            # 时间步+1
            time_step += 1



if __name__ == '__main__':
    animation = Animation_robot()
    animation.fig_set()

    controller = Main_controller()

    traj_x, traj_y, traj_th, \
    traj_g_x, traj_g_y, traj_paths, \
    traj_opt, obstacles = controller.run_to_goal()

