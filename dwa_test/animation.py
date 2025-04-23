import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import math
import sys

# 绘制圆形
def write_circle(center_x, center_y, angle, circle_size=0.2):  # 默认圆形大小(半径0.2米，相当于人的大小约15cm半径)
    """
    绘制圆形及其方向线
    参数:
        center_x, center_y - 圆心坐标
        angle - 方向角度(弧度)
        circle_size - 圆半径大小
    返回:
        圆形x坐标列表, 圆形y坐标列表, 方向线x坐标, 方向线y坐标
    """
    # 初始化
    circle_x = []  # 圆形x坐标列表
    circle_y = []  # 圆形y坐标列表

    steps = 100  # 圆形绘制的分辨率(点数)
    for i in range(steps):
        # 计算圆周上的点坐标
        circle_x.append(center_x + circle_size * math.cos(i * 2 * math.pi / steps))
        circle_y.append(center_y + circle_size * math.sin(i * 2 * math.pi / steps))
    
    # 计算方向线坐标(从圆心指向角度方向)
    circle_line_x = [center_x, center_x + math.cos(angle) * circle_size]
    circle_line_y = [center_y, center_y + math.sin(angle) * circle_size]
    
    return circle_x, circle_y, circle_line_x, circle_line_y

class Path_anim():
    """路径动画类，用于绘制DWA算法生成的候选路径"""
    def __init__(self, axis):
        # 初始化路径图像(青色虚线)
        self.path_img, = axis.plot([], [], color='c', linestyle='dashed', linewidth=0.15)

    def set_graph_data(self, x, y):
        """设置路径数据"""
        self.path_img.set_data(x, y)
        return self.path_img, 
    
class Obstacle_anim():
    """障碍物动画类，用于绘制障碍物"""
    def __init__(self, axis):
        # 初始化障碍物图像(黑色实线)
        self.obs_img, = axis.plot([], [], color='k')

    def set_graph_data(self, obstacle):
        """设置障碍物数据"""
        angle = 0.0  # 障碍物不需要方向指示
        # 绘制圆形障碍物
        circle_x, circle_y, circle_line_x, circle_line_y = \
                write_circle(obstacle.x, obstacle.y, angle, circle_size=obstacle.size)

        self.obs_img.set_data(circle_x, circle_y)
        return self.obs_img, 

class Animation_robot():
    """机器人动画主类，负责整体动画的绘制和控制"""
    def __init__(self):
        # 创建图形窗口
        self.fig = plt.figure()
        # 添加子图
        self.axis = self.fig.add_subplot(111)

    def fig_set(self):
        """图形窗口初始设置"""
        # 设置坐标轴范围
        MAX_x = 12
        min_x = -12
        MAX_y = 12
        min_y = -12

        self.axis.set_xlim(min_x, MAX_x)
        self.axis.set_ylim(min_y, MAX_y)

        # 显示网格
        self.axis.grid(True)

        # 设置纵横比相等(保证圆形不变形)
        self.axis.set_aspect('equal')

        # 设置坐标轴标签
        self.axis.set_xlabel('X [m]')
        self.axis.set_ylabel('Y [m]')

    def plot(self, traj_x, traj_y):
        """静态轨迹绘制方法"""
        self.axis.plot(traj_x, traj_y)
        plt.show()
    
    def func_anim_plot(self, traj_x, traj_y, traj_th, traj_paths, traj_g_x, traj_g_y, traj_opt, obstacles):
        """
        动态动画绘制方法
        参数:
            traj_x, traj_y - 机器人轨迹坐标列表
            traj_th - 机器人朝向角度列表
            traj_paths - DWA算法生成的候选路径列表
            traj_g_x, traj_g_y - 目标点坐标列表
            traj_opt - 最优轨迹列表
            obstacles - 障碍物列表
        """
        # 保存轨迹数据到实例变量
        self.traj_x = traj_x
        self.traj_y = traj_y
        self.traj_th = traj_th
        self.traj_paths = traj_paths
        self.traj_g_x = traj_g_x
        self.traj_g_y = traj_g_y
        self.traj_opt = traj_opt
        self.obstacles = obstacles

        # 初始化轨迹线(黑色虚线)
        self.traj_img, = self.axis.plot([], [], 'k', linestyle='dashed')

        # 初始化机器人圆形和方向线
        self.robot_img, = self.axis.plot([], [], 'k')  # 机器人圆形
        self.robot_angle_img, = self.axis.plot([], [], 'k')  # 方向线

        # 初始化目标点标记(蓝色星号)
        self.img_goal, = self.axis.plot([], [], '*', color='b', markersize=15)

        # 初始化DWA候选路径(最多100条)
        self.dwa_paths = []
        self.max_path_num = 100
        for k in range(self.max_path_num):
            self.dwa_paths.append(Path_anim(self.axis))

        # 初始化最优轨迹线(红色虚线)
        self.traj_opt_img, = self.axis.plot([], [], 'r', linestyle='dashed')

        # 初始化障碍物
        self.obs = []
        self.obstacles_num = len(obstacles)
        for k in range(len(obstacles)):
            self.obs.append(Obstacle_anim(self.axis))
        
        # 添加步数文本显示
        self.step_text = self.axis.text(0.05, 0.9, '', transform=self.axis.transAxes)

        # 创建动画(每帧间隔100ms)
        animation = ani.FuncAnimation(self.fig, self._update_anim, interval=100, \
                              frames=len(traj_g_x))
                              
        # 注释掉的动画保存功能
        # print('save_animation?')
        # shuold_save_animation = int(input())
        # if shuold_save_animation: 
        #     animation.save('basic_animation.gif', writer='imagemagick')

        plt.show()

    def _update_anim(self, i):
        """动画更新函数(每帧调用)"""
        # 初始化图像列表
        self.dwa_imgs = []
        self.dwa_path_imgs = []
        self.obs_imgs = []

        # 更新轨迹线数据
        self.traj_img.set_data(self.traj_x[:i+1], self.traj_y[:i+1])

        # 绘制当前机器人位置和方向
        circle_x, circle_y, circle_line_x, circle_line_y = write_circle(
            self.traj_x[i], self.traj_y[i], self.traj_th[i], circle_size=0.2)
        self.robot_img.set_data(circle_x, circle_y)
        self.robot_angle_img.set_data(circle_line_x, circle_line_y)

        # 更新目标点位置
        self.img_goal.set_data([self.traj_g_x[i]], [self.traj_g_y[i]])

        # 更新最优轨迹
        self.traj_opt_img.set_data(self.traj_opt[i].x, self.traj_opt[i].y)

        # 更新DWA候选路径(从所有候选路径中均匀选取max_path_num条显示)
        count = 0
        for k in range(self.max_path_num):
            path_num = math.ceil(len(self.traj_paths[i])/(self.max_path_num)) * k
            
            if path_num > len(self.traj_paths[i]) - 1:
                path_num = np.random.randint(0, len(self.traj_paths[i]))

            self.dwa_path_imgs.append(
                self.dwa_paths[k].set_graph_data(
                    self.traj_paths[i][path_num].x, 
                    self.traj_paths[i][path_num].y))

        # 更新障碍物位置
        for k in range(self.obstacles_num):
            self.obs_imgs.append(self.obs[k].set_graph_data(self.obstacles[k]))      

        # 更新步数文本
        self.step_text.set_text('step = {0}'.format(i))

        # 收集所有需要更新的图像对象
        for img in [self.traj_img, self.robot_img, self.robot_angle_img, 
                   self.img_goal, self.step_text, self.dwa_path_imgs, 
                   self.obs_imgs, self.traj_opt_img]:
            self.dwa_imgs.append(img)

        return self.dwa_imgs
