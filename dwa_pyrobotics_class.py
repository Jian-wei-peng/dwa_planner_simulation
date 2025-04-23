import numpy as np
import math
import matplotlib.pyplot as plt

from enum import Enum


def diff_model(state, cmd, dt):
  """
  差速移动机器人运动学模型
  Args:
      state: 机器人当前状态--[x,y,yaw,v,w]
      cmd: 控制指令--[v, w]
      dt: 采样周期
  Returns:
      state: 更新后的状态 --- x, y, yaw, v, w
  """

  state[0] += cmd[0] * math.cos(state[2]) * dt
  state[1] += cmd[0] * math.sin(state[2]) * dt
  state[2] += cmd[1] * dt

  state[3] = cmd[0]
  state[4] = cmd[1]

  return state


class RobotType(Enum):
    circle = 0
    rectangle = 1


class Config:
    
    def __init__(self):

        # 采样周期 [s]
        self.dt = 0.1

        # 轨迹推算时间 [s]
        self.predict_time = 3.0

        # 机器人几何尺寸 [m]
        self.robot_radius = 1.0
        self.robot_width  = 0.5 
        self.robot_length = 1.2

        # circle = 0;   rectangle = 1
        self.robot_type = RobotType.circle

        # 若与障碍物的最小距离大于阈值（例如这里设置的阈值为robot_radius+0.2）,则设为一个较大的常值
        self.judge_distance = 10

        # 线速度边界 [m/s]
        self.v_max = 1.0
        self.v_min = -0.5

        # 角速度边界 [rad/s]
        self.w_max = 40.0 * math.pi / 180.0
        self.w_min = -40.0 * math.pi / 180.0

        # 线加速度和角加速度最大值
        self.a_v_max = 0.2   # m/ss
        self.a_w_max = 40.0 * math.pi / 180.0    # 40 rad/ss

        # 采样分辨率, 在速度空间中搜索的步长
        self.v_sample = 0.01 # m/s
        self.w_sample = 0.1 * math.pi / 180.0 # 0.1 rad

        # 轨迹评价函数权重
        self.w_h = 0.15
        self.w_v = 1.0
        self.w_o = 1.0

        # 目标点位置 [x(m), y(m)]
        self.target = np.array([10, 10])

        # 障碍物位置 [x(m), y(m)], dim: [num_ob,2]
        self.obstacles = np.array([[-1, -1],
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
        # 障碍物安全膨胀范围半径
        self.obstacle_radius = 0.2

        # 防止机器人运动过程卡死
        self.robot_stuck_flag = 0.001
        
    # @property 装饰器用于将一个方法转换为一个只读属性。当你访问这个属性时，Python 会自动调用这个方法，并返回其结果
    # robot_type 方法被转换为一个只读属性。当你访问 config.robot_type 时，Python 会调用 robot_type 方法，并返回 self._robot_type 的值
    @property
    def robot_type(self):
        return self._robot_type
    
    # @robot_type.setter 装饰器用于定义一个属性的设置器。当你尝试给这个属性赋值时，Python 会自动调用这个设置器方法
    # robot_type 方法被转换为一个设置器
    # 当你尝试给 config.robot_type 赋值时，Python 会调用 robot_type 方法，并将新值传递给 value 参数
    # 在这个方法中，首先会检查 value 是否是 RobotType 的实例，如果不是，则会抛出一个 TypeError 异常
    # 如果检查通过，则将 value 赋值给 self._robot_type
    @robot_type.setter 
    def robot_type(self, value):
        if not isinstance(value, RobotType):
            raise TypeError("robot_type must be an instance of RobotType")
        self._robot_type = value


class DWA:
    def __init__(self, config, robot_type) -> None:
        """
        构造函数
        Args:
            config: 参数类
        """
        self.dt = config.dt
        self.predict_time = config.predict_time

        self.v_min = config.v_min
        self.w_min = config.w_min
        self.v_max = config.v_max
        self.w_max = config.w_max

        self.a_v_max = config.a_v_max
        self.a_w_max = config.a_w_max

        self.v_sample = config.v_sample
        self.w_sample = config.w_sample

        self.w_h  = config.w_h
        self.w_v  = config.w_v
        self.w_o = config.w_o

        self.robot_radius = config.robot_radius
        self.robot_width  = config.robot_width
        self.robot_length = config.robot_length

        self.obstacle_radius = config.obstacle_radius

        self.robot_stuck_flag = config.robot_stuck_flag

        self.robot_type = robot_type

    def dwa_control(self, state_cur, goal, obstacles):
        """
        DWA规划控制器
        Args:
            state_cur: 机器人当前状态--[x, y, yaw, v, w]
            goal: 目标点位置--[x, y]
            obstacles: 障碍物位置--[x, y], dim: [num_ob,2]
        Returns:
            cmd: DWA输出的最优控制指令 [v, w]
            trajectory: 最优控制指令对应的轨迹 [x, y]
        """

        cmd, trajectory = self.trajectory_evaluation(state_cur, goal, obstacles)

        return cmd, trajectory

    def trajectory_evaluation(self, state_cur, goal, obstacles):
        """
        轨迹评价函数, 评价越高, 轨迹越优

        Args:
            state_cur: 机器人当前状态--[x, y, yaw, v, w]
            dynamic_window: 速度空间---[v_low,v_high,w_low,w_high]
            goal: 目标点位置--[x, y]
            obstacles: 障碍物位置--[x, y], dim: [num_ob,2]

        Returns:
            cmd_opt: 最优控制量
            trajectory_opt: 最优轨迹
        """

        # 最优代价, 初始化为无穷大
        cost_min = float('inf')

        # 最优控制量, 初始化为0
        cmd_opt = [0.0, 0.0]

        # 最优轨迹, 初始化为当前状态[x, y, yaw, v, w]
        trajectory_opt = state_cur

        # 1. 构建速度空间
        allowable_V, allowable_W = self._get_dynamic_windows(state_cur)

        # 2. 速度采样
        for v in np.arange(allowable_V[0], allowable_V[1], self.v_sample):
            for w in np.arange(allowable_W[0], allowable_W[1], self.w_sample):
                
                # 3. 轨迹推算 [x, y, yaw, v, w]
                trajectory = self._trajectory_predict(state_cur, v, w)

                # 4. 计算推算轨迹的代价
                heading  = self.w_h * self._heading_cost(trajectory, goal)
                dist     = self.w_o * self._dist_cost(trajectory, obstacles)
                velocity = self.w_v * (self.v_max - trajectory[-1, 3])

                cost_total = heading + dist + velocity

                # 保留总代价最小的轨迹及对应的采样速度, 作为最优输出
                if cost_min >= cost_total:
                    cost_min = cost_total
                    trajectory_opt = trajectory
                    cmd_opt = [v, w]

                    # to ensure the robot do not get stuck in
                    if abs(cmd_opt[0]) < self.robot_stuck_flag and abs(state_cur[3]) < self.robot_stuck_flag:
                        cmd_opt[1] = -self.a_w_max

        return cmd_opt, trajectory_opt

    def _get_dynamic_windows(self, state_cur):
        """
        计算速度空间窗口

        Args:
            state_cur: 机器人当前状态--[x, y, yaw, v, w]
        Returns:
            [v_low,v_high,w_low,w_high]: 最终采样后的速度空间
        """

        # 速度边界限制
        Vs = [self.v_min, self.v_max, self.w_min, self.w_max]

        # 加速度限制, state_cur[3]和state_cur[4]是机器人当前速度
        Vd = [state_cur[3] - self.a_v_max * self.dt,
              state_cur[3] + self.a_v_max * self.dt,
              state_cur[4] - self.a_w_max * self.dt,
              state_cur[4] + self.a_w_max * self.dt ]
        
        # 理论上还有一个障碍物限制, 但可直接先使用Vs和Vd的交集的速度组合采样推算出轨迹后
        # 计算当前速度下对应预测轨迹与障碍物之间的最近距离，然后看当前采样的这对速度能否在碰到障碍物之前停下 
        # 如果可以，则这对速度是可接受的，否则需要抛弃掉

        # 搜索空间
        allowable_V = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1])]
        allowable_W = [max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

        return allowable_V, allowable_W

    def _trajectory_predict(self, state_cur, v, w):
        """
        根据采样速度和机器人当前位置进行轨迹推算
        Args:
            state_cur: 机器人当前状态--[x, y, yaw, v, w]
            v: 当前采样的线速度
            w: 当前采样的角速度
        Returns:
            trajectory: 推算出的轨迹
        """

        state = np.array(state_cur)
        trajectory = state
        time = 0

        # 在预测周期内进行轨迹推算
        while time <= self.predict_time:
            x = diff_model(state, [v, w], self.dt)
            trajectory = np.vstack((trajectory, x))
            time += self.dt

        return trajectory

    def _heading_cost(self, trajectory, goal):
        """
        方位角代价函数: 评估在当前采样速度下产生的轨迹终点位置朝向与目标点连线夹角的误差
        Args:
            trajectory: 当前采样速度产生的轨迹 [x, y, yaw, v, w]
            goal: 目标点位置--[x, y]
        Returns:
            朝向角偏差代价函数值
        """

        # 轨迹末端点到目标点的位置偏差
        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]

        # 计算目标点与轨迹末端点连线的角度 [rad]
        angle_h2r = math.atan2(dy, dx)

        # 将目标点与轨迹末端连线的角度 angle_h2r 与轨迹末端点的朝向角 trajectory[-1, 2] 之间的差值作为朝向角代价
        return abs( math.atan2( math.sin(angle_h2r - trajectory[-1, 2]), math.cos(angle_h2r - trajectory[-1, 2]) ) )

    def _dist_cost(self, trajectory, obstacles):
        """
        与障碍物间距的评价函数：在当前采样速度下产生的轨迹与障碍物之间的最近距离
        Args:
            trajectory: 当前采样速度产生的轨迹 [x, y, yaw, v, w]
            obstacles: 障碍物位置--[x, y], dim: [num_ob,2]
        Returns:
            dist_cost: 与障碍物间距的代价函数值
        """

        # 提取所有障碍物的 x 和 y 坐标
        ox = obstacles[:, 0]
        oy = obstacles[:, 1]

        # 计算轨迹中每个点的 x 和 y 坐标与所有障碍物的 x 和 y 坐标的差值
        dx = trajectory[:, 0] - ox[:, None]
        dy = trajectory[:, 1] - oy[:, None]

        # 计算轨迹中每个点到所有障碍物的欧几里得距离
        r = np.hypot(dx, dy)

        # 检查机器人的边界是否与障碍物发生碰撞
        if self.robot_type == RobotType.rectangle:

            # 提取轨迹中每个点的朝向角
            yaw = trajectory[:, 2]

            # 计算每个点的旋转矩阵, 将障碍物坐标转换到机器人的局部坐标系
            rot_matrix = np.array([[np.cos(yaw), -np.sin(yaw)], 
                                    [np.sin(yaw), np.cos(yaw)]])
            rot_matrix = np.transpose(rot_matrix, [2, 0, 1])

            # 计算每个障碍物和轨迹上每个点的位置偏移量
            ob_in_robot = obstacles[:, None] - trajectory[:, 0:2]
            ob_in_robot = ob_in_robot.reshape(-1, ob_in_robot.shape[-1])

            # 将偏移量转换到机器人坐标系下
            ob_in_robot = np.array([ob_in_robot @ x for x in rot_matrix])
            ob_in_robot = ob_in_robot.reshape(-1, ob_in_robot.shape[-1])

            # 边界检查
            upper_check  = ob_in_robot[:, 0] <= self.robot_length / 2
            bottom_check = ob_in_robot[:, 0] >= -self.robot_length / 2
            right_check  = ob_in_robot[:, 1] <= self.robot_width / 2
            left_check   = ob_in_robot[:, 1] >= -self.robot_width / 2 

            # 如果任何一个障碍物在机器人的边界内, 则返回无穷大(Inf), 表示发生碰撞
            if ( np.logical_and( np.logical_and(upper_check, right_check),
                                 np.logical_and(bottom_check, left_check) ) ).any():
                return float("Inf")

        elif self.robot_type == RobotType.circle:
            # 如果轨迹中存在任意一点距离障碍物小于安全阈值(机器人半径 + 障碍物膨胀半径), 则返回无穷大(Inf), 表示发生碰撞
            if np.array( r <= config.robot_radius + self.obstacle_radius ).any():
                return float("Inf")

        return 1.0 / np.min(r)


# 绘制一个箭头, 表示机器人朝向
def plot_arrow(x, y, yaw, length = 0.5, width = 0.1):
    # x, y: 箭头的起点坐标; yaw: 箭头的朝向角度（以弧度表示）; length: 箭头的长度，默认值为 0.5; width: 箭头的头部宽度，默认值为 0.1
    # length * math.cos(yaw), length * math.sin(yaw): 箭头的终点相对于起点的偏移量，根据 yaw 角度和 length 计算得出
    # head_length: 箭头头部的长度; head_width: 箭头头部的宽度
    plt.arrow(x, 
              y, 
              length * math.cos(yaw), 
              length * math.sin(yaw),
              head_length = width, 
              head_width = width
              )
    # 在箭头的起点位置绘制一个点，用于标记箭头的起点
    plt.plot(x, y)


# 绘制一个机器人，包括机器人的几何轮廓和朝向线
def plot_robot(x, y, yaw, config):
    if config.robot_type == RobotType.rectangle:
        outline = np.array([[-config.robot_length / 2, config.robot_length / 2,
                             config.robot_length / 2, -config.robot_length / 2,
                             -config.robot_length / 2], 
                            [config.robot_width / 2,   config.robot_width / 2,
                             -config.robot_width / 2, -config.robot_width / 2,
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


def main(config, robot_type):
    
    print(__file__ + " start!!")

    # 机器人初始状态 [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    x = np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0])

    # 目标点位置 [x(m), y(m)]
    goal = config.target

    # 障碍物位置
    ob = config.obstacles

    # 机器人类型，默认为圆
    config.robot_type = robot_type

    # 轨迹
    trajectory = np.array(x)

    # 实例化DWA
    dwa = DWA(config, robot_type)

    while True:
        # 调用DWA, 得到机器人控制指令和最优轨迹
        u, predicted_trajectory = dwa.dwa_control(x, goal, ob)

        # 利用运动学模型更新机器人状态
        x = diff_model(x, u, config.dt)

        # 记录轨迹
        # np.vstack是按 行方向（垂直方向） 合并数组，动态扩展轨迹数据
        trajectory = np.vstack((trajectory, x))

        # 绘图
        # 清除图形内容
        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        # 绿色实线绘制预测轨迹
        plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")
        # 红色叉号标记机器人当前位置
        plt.plot(x[0], x[1], "xr")
        # 蓝色叉号标记目标位置
        plt.plot(goal[0], goal[1], "xb")
        # 黑色圆圈标记障碍物位置
        plt.plot(ob[:, 0], ob[:, 1], "ok")
        # 绘制机器人的形状
        plot_robot(x[0], x[1], x[2], config)
        # 绘制机器人方向的箭头
        plot_arrow(x[0], x[1], x[2])
        # 设置 x 轴和 y 轴的比例相同
        plt.axis("equal")
        # 显示网格线
        plt.grid(True)
        # 程序暂停 0.001 秒（即 1 毫秒），然后继续执行后续代码。这个函数通常用于动态更新图形时，控制图形的刷新频率
        plt.pause(0.001)

        # 机器人到目标点的位置
        dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])

        # 机器人位置与目标点距离小于机器人半径即到达目标点
        if robot_type == RobotType.circle and dist_to_goal <= config.robot_radius:
            print("Goal!!!")
            break
        elif robot_type == RobotType.rectangle and dist_to_goal <= math.hypot(config.robot_length / 2, config.robot_width / 2):
            print("Goal!!!")
            break
    
    # 仿真结束
    print("Done")
    # 红色实线绘制机器人轨迹
    plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")
    plt.pause(0.001)
    # 显示当前绘制的图形
    plt.show()


if __name__ == '__main__':

    config = Config()

    # main(config, RobotType.circle)
    main(config, RobotType.rectangle)



