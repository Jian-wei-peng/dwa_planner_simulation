import numpy as np
np.seterr(divide='ignore', invalid='ignore')

from animation import Animation_robot
import math

# 基本函数
# 正规化函数，将数据按最大最小值进行归一化处理

# def min_max_normalize(data):
#     data = np.array(data)
#     min_val, max_val = np.min(data), np.max(data)
#     if max_val == min_val:
#         return np.zeros_like(data)
#     return (data - min_val) / (max_val - min_val)

def min_max_normalize(data):
    data = np.array(data)
    
    max_data = max(data)
    min_data = min(data)

    # 如果最大值和最小值相等（即数据没有变化），将数据归一化为0
    if max_data - min_data == 0:
        data = [0.0 for i in range(len(data))]
    else:
        data = (data - min_data) / (max_data - min_data)

    return data

# 角度范围修正函数，确保角度在 -π 到 π 之间
def angle_range_corrector(angle):
    if angle > math.pi:
        while angle > math.pi:
            angle -= 2 * math.pi
    elif angle < -math.pi:
        while angle < -math.pi:
            angle += 2 * math.pi
    
    return angle

# # 绘制圆形轨迹
# def write_circle(center_x, center_y, angle, circle_size=0.2):  # 人的大小为半径15cm
#     circle_x = []  # 用于保存圆形的 x 坐标
#     circle_y = []  # 用于保存圆形的 y 坐标

#     steps = 100  # 圆的分解度，100步足够精确
#     for i in range(steps):
#         # 根据圆的公式计算圆上的各个点
#         circle_x.append(center_x + circle_size * math.cos(i * 2 * math.pi / steps))
#         circle_y.append(center_y + circle_size * math.sin(i * 2 * math.pi / steps))
    
#     # 绘制圆形的指示线（朝向指定角度的线段）
#     circle_line_x = [center_x, center_x + math.cos(angle) * circle_size]
#     circle_line_y = [center_y, center_y + math.sin(angle) * circle_size]
    
#     return circle_x, circle_y, circle_line_x, circle_line_y

# 规则说明：
# x, y, th 是机器人当前的状态
# g_ 是目标位置
# traj_ 是机器人过去的轨迹
# 单位：角度使用弧度（rad），位置使用米（m）
# 由于机器人是二轮驱动，输入包括速度和角速度

# 定义路径类（用于存储机器人的路径）
class Path():
    def __init__(self, u_th, u_v): 
        self.x = None
        self.y = None
        self.th = None
        self.u_v = u_v  # 线速度
        self.u_th = u_th  # 角速度

# 定义障碍物类（用于存储障碍物信息）
class Obstacle():
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size
        
# 定义二轮机器人类（模拟机器人行为）
class Two_wheeled_robot():
    def __init__(self, init_x, init_y, init_th):
        # 初始化机器人状态
        self.x = init_x
        self.y = init_y
        self.th = init_th
        self.u_v = 0.0
        self.u_th = 0.0

        # 存储机器人的轨迹
        self.traj_x = [init_x]
        self.traj_y = [init_y]
        self.traj_th = [init_th]
        self.traj_u_v = [0.0]
        self.traj_u_th = [0.0]

    # 更新机器人状态
    def update_state(self, u_th, u_v, dt):  
        self.u_th = u_th
        self.u_v = u_v
        
        # 根据速度和角速度计算下一状态
        next_x = self.u_v * math.cos(self.th) * dt + self.x
        next_y = self.u_v * math.sin(self.th) * dt + self.y
        next_th = self.u_th * dt + self.th

        # 更新轨迹
        self.traj_x.append(next_x)
        self.traj_y.append(next_y)
        self.traj_th.append(next_th)

        # 更新状态
        self.x = next_x
        self.y = next_y
        self.th = next_th

        return self.x, self.y, self.th

# 定义 DWA 仿真器（用于预测机器人状态）
class Simulator_DWA_robot():
    def __init__(self):
        self.max_accelation = 1.0  # 最大加速度
        self.max_ang_accelation = 100 * math.pi / 180  # 最大角加速度
        self.lim_max_velo = 1.6  # 最大线速度（1.6 m/s）
        self.lim_min_velo = 0.0  # 最小线速度（0 m/s）
        self.lim_max_ang_velo = math.pi  # 最大角速度（π rad/s）
        self.lim_min_ang_velo = -math.pi  # 最小角速度（-π rad/s）

    # 预测未来的状态
    def predict_state(self, ang_velo, velo, x, y, th, dt, pre_step):
        next_xs = []
        next_ys = []
        next_ths = []

        for i in range(pre_step):
            temp_x = velo * math.cos(th) * dt + x
            temp_y = velo * math.sin(th) * dt + y
            temp_th = ang_velo * dt + th

            next_xs.append(temp_x)
            next_ys.append(temp_y)
            next_ths.append(temp_th)

            x = temp_x
            y = temp_y
            th = temp_th

        return next_xs, next_ys, next_ths

# DWA（动态窗口法）路径规划类
class DWA():
    def __init__(self):
        self.simu_robot = Simulator_DWA_robot()
        self.pre_time = 3  # 预测时间（秒）
        self.pre_step = 30  # 预测步数
        self.delta_velo = 0.02  # 速度增量
        self.delta_ang_velo = 0.02  # 角速度增量
        self.samplingtime = 0.1  # 采样时间
        self.weight_angle = 0.04  # 角度权重
        self.weight_velo = 0.2  # 速度权重
        self.weight_obs = 0.1  # 障碍物权重
        self.traj_paths = []  # 存储所有路径
        self.traj_opt = []  # 存储最优路径

    # 计算控制输入，生成路径并评估
    def calc_input(self, g_x, g_y, state, obstacles):
        paths = self._make_path(state)  # 生成所有路径
        opt_path = self._eval_path(paths, g_x, g_y, state, obstacles)  # 评估路径并选择最优路径
        self.traj_opt.append(opt_path)
        
        return paths, opt_path

    # 生成路径
    def _make_path(self, state):
        min_ang_velo, max_ang_velo, min_velo, max_velo = self._calc_range_velos(state)  # 计算角速度和线速度的范围
        paths = []  # 存储所有路径

        for ang_velo in np.arange(min_ang_velo, max_ang_velo, self.delta_ang_velo):
            for velo in np.arange(min_velo, max_velo, self.delta_velo):
                path = Path(ang_velo, velo)
                next_x, next_y, next_th = self.simu_robot.predict_state(ang_velo, velo, state.x, state.y, state.th, self.samplingtime, self.pre_step)
                path.x = next_x
                path.y = next_y
                path.th = next_th
                paths.append(path)

        self.traj_paths.append(paths)  # 保存所有路径
        return paths

    # 计算速度范围
    def _calc_range_velos(self, state):
        range_ang_velo = self.samplingtime * self.simu_robot.max_ang_accelation
        min_ang_velo = state.u_th - range_ang_velo
        max_ang_velo = state.u_th + range_ang_velo
        if min_ang_velo < self.simu_robot.lim_min_ang_velo:
            min_ang_velo = self.simu_robot.lim_min_ang_velo
        if max_ang_velo > self.simu_robot.lim_max_ang_velo:
            max_ang_velo = self.simu_robot.lim_max_ang_velo

        range_velo = self.samplingtime * self.simu_robot.max_accelation
        min_velo = state.u_v - range_velo
        max_velo = state.u_v + range_velo
        if min_velo < self.simu_robot.lim_min_velo:
            min_velo = self.simu_robot.lim_min_velo
        if max_velo > self.simu_robot.lim_max_velo:
            max_velo = self.simu_robot.lim_max_velo

        return min_ang_velo, max_ang_velo, min_velo, max_velo

    # 评估路径，选择最优路径
    def _eval_path(self, paths, g_x, g_y, state, obstacles):
        nearest_obs = self._calc_nearest_obs(state, obstacles)  # 获取离机器人最近的障碍物
        score_heading_angles = []
        score_heading_velos = []
        score_obstacles = []

        for path in paths:
            score_heading_angles.append(self._heading_angle(path, g_x, g_y))  # 评估角度误差
            score_heading_velos.append(self._heading_velo(path))  # 评估速度
            score_obstacles.append(self._obstacle(path, nearest_obs))  # 评估障碍物

        # 正规化得分
        for scores in [score_heading_angles, score_heading_velos, score_obstacles]:
            scores = min_max_normalize(scores)

        # score_heading_angles = min_max_normalize(score_heading_angles)
        # score_heading_velos = min_max_normalize(score_heading_velos)
        # score_obstacles = min_max_normalize(score_obstacles)

        score = 0.0
        # score = -float('inf')
        for k in range(len(paths)):
            temp_score = self.weight_angle * score_heading_angles[k] + \
                         self.weight_velo * score_heading_velos[k] + \
                         self.weight_obs * score_obstacles[k]
        
            if temp_score > score:
                opt_path = paths[k]
                score = temp_score
                
        return opt_path

    # 评估路径的角度误差
    def _heading_angle(self, path, g_x, g_y):
        last_x = path.x[-1]
        last_y = path.y[-1]
        last_th = path.th[-1]

        angle_to_goal = math.atan2(g_y - last_y, g_x - last_x)
        score_angle = angle_to_goal - last_th
        score_angle = abs(angle_range_corrector(score_angle))
        score_angle = math.pi - score_angle

        return score_angle

    # 评估路径的速度
    def _heading_velo(self, path):
        return path.u_v

    # 计算障碍物
    def _calc_nearest_obs(self, state, obstacles):
        area_dis_to_obs = 5  # 考虑的障碍物区域范围
        nearest_obs = []

        for obs in obstacles:
            temp_dis_to_obs = math.sqrt((state.x - obs.x) ** 2 + (state.y - obs.y) ** 2)
            if temp_dis_to_obs < area_dis_to_obs:
                nearest_obs.append(obs)

        return nearest_obs

    # 评估路径与障碍物的碰撞情况
    def _obstacle(self, path, nearest_obs):
        score_obstacle = 2
        temp_dis_to_obs = 0.0

        for i in range(len(path.x)):
            for obs in nearest_obs: 
                temp_dis_to_obs = math.sqrt((path.x[i] - obs.x) ** 2 + (path.y[i] - obs.y) ** 2)
                
                if temp_dis_to_obs < score_obstacle:
                    score_obstacle = temp_dis_to_obs  # 最接近障碍物的距离

                if temp_dis_to_obs < obs.size + 0.75:  # 如果碰到障碍物，给路径一个负分
                    score_obstacle = -float('inf')
                    break
            
            else:
                continue
            
            break

        return score_obstacle

# 目标生成类
class Const_goal():
    def __init__(self):
        self.traj_g_x = []
        self.traj_g_y = []

    def calc_goal(self, time_step):
        if time_step <= 100:
            g_x = 10.0
            g_y = 10.0
        else:
            g_x = -10.0
            g_y = -10.0

        self.traj_g_x.append(g_x)
        self.traj_g_y.append(g_y)

        return g_x, g_y

# 主控制类
class Main_controller():
    def __init__(self):

        self.samplingtime = 0.1

        self.robot = Two_wheeled_robot(0.0, 0.0, 0.0)

        self.goal_maker = Const_goal()

        self.obstacles = [Obstacle(4, 1, 0.25), Obstacle(0, 4.5, 0.25), 
                          Obstacle(3, 4.5, 0.25), Obstacle(5, 3.5, 0.25), 
                          Obstacle(7.5, 9.0, 0.25)]
        
        self.controller = DWA()


    def run_to_goal(self):
        goal_flag = False
        time_step = 0

        while not goal_flag:
            g_x, g_y = self.goal_maker.calc_goal(time_step)
            paths, opt_path = self.controller.calc_input(g_x, g_y, self.robot, self.obstacles)

            u_th = opt_path.u_th
            u_v = opt_path.u_v

            self.robot.update_state(u_th, u_v, self.samplingtime)

            dis_to_goal = np.sqrt((g_x - self.robot.x)**2 + (g_y - self.robot.y)**2)
            if dis_to_goal < 0.5:  # 判断是否到达目标
                goal_flag = True
            
            time_step += 1

        return self.robot.traj_x, self.robot.traj_y, self.robot.traj_th, \
               self.goal_maker.traj_g_x, self.goal_maker.traj_g_y, \
               self.controller.traj_paths, self.controller.traj_opt, self.obstacles

def main():
    animation = Animation_robot()
    animation.fig_set()

    controller = Main_controller()
    traj_x, traj_y, traj_th, traj_g_x, traj_g_y, traj_paths, traj_opt, obstacles = controller.run_to_goal()

    ani = animation.func_anim_plot(traj_x, traj_y, traj_th, traj_paths, traj_g_x, traj_g_y, traj_opt, obstacles)

if __name__ == '__main__':
    main()
