"""
功能：模拟平面波、球面波、干涉等基础光学现象
参数：波矢k，位矢r
输出：光场分布
作者：202128013920003刘夕铭
机构：中国科学院长春光学机密机械与物理研究所
时间：2022.03.18@9.13
说明：（1）针对球面波：说明：对于e（x,y,z）=A/r*cos(kr-wt)，[E,X,Y,Z]作为4元数（暂时不考虑时间）的可视化，对本人可能比较难以实现
     根据哔哩哔哩网站中博主“3B1B”的“四元数可视化”的视频，对于此仿真做一些说明。
     https://www.bilibili.com/video/BV1SW411y7W1?from=search&seid=16927732778844735999&spm_id_from=333.337.0.0
     假设是发散波，以E的最大值作归一化变量e∈[0,E]垂直于x,y,z的空间正交基的方向作为4D超球，
     对约束为“E^2+X^2+Y^2+Z^2=1"的4D超球进行3D投影，3D中心原点为E,坐标轴为X,Y,Z
     球面方程为“x^2+y^2+z^2=1-e^2”，当e在增减时，其4D超球的尺度不变，而球面方程在不断缩放
     在光波传播过程中，（在此考虑时间）对于一个周期T时间段内，其振幅相对于周期时间中心T/2以cos规律变化，
     对于周期T时间平均值，以1/r不断衰减。最终在3D中呈现出以光波产生点强度e^2为最大E，
     随着传播e逐步下降，球面越来越大，并且球面变化以类简谐方式运动“x^2+y^2+z^2=1-[A/r*cos(kr-wt)]^2”
     现在球面波变为新4元数[T,X,Y,Z]的方程；；；；；；；若不考虑T，此时函数是[x,y,z]的隐函数
     在matlab中，从2016b版本更新了函数fimplicit3可以对3D隐函数进行绘图，但源码无法直接改写实现。
                也包括isosurface命令画出隐函数的等相面，在报告中会给出。
     求解隐函数的方法在Python中，有fsolve和root法，但对于类球形解算不友好，只对凸函数进行解算。
     由于模拟球对称函数，所以考虑使用r特征代替[x,y,z]，再加上T变量表示等相面
               总而言之，想三维表示球面波，对本人能力来说是够呛了
          于是根据《物理光学》第四版.梁铨廷著。书中（1.43）进行了程序编写
     最后更改时间：2022.03.17@9:24
    （2）针对平面波：以二维平面波进行模拟，突出振幅与空间的分布
     最后更改时间：2022.03.16@15:35
    （3）针对干涉：以二维平面波进行模拟，突出振幅与空间的分布，取两光束所在平面为谈论平面,取两光振幅都为A,波矢模为k
     E1(x,y)=A*cos(k_x * x + k_y * y)
     E2(x,y)=A*cos(k_x * x + k_y * y)
     E（x,y）= E1(x,y) + E2(x,y) = 2 * A * cos(cos(k_x * x)) * cos(sin(k_y *y))
     最后更改时间：时间：2022.03.17@14.38
"""

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


class OpticalBasicSimulation:

    def __init__(self, method=None, A=1, c=1, t=1, k=None, xgap=100, ygap=100, xextent=8, yextent=8):
        if k is None:
            k = [1, 1]
        self.method = method
        self.A = A
        self.c = c
        self.t = t
        self.k = k
        self.xgap = xgap
        self.ygap = ygap
        self.xextent = xextent
        self.yextent = yextent
        self.x_Vector = np.linspace(-self.xextent, self.xextent, self.xgap)
        self.y_Vector = np.linspace(-self.yextent, self.yextent, self.ygap)  # 在范围（-10,10）之间，以x_gap，y_gap采样
        self.r_xGrid, self.r_yGrid = np.meshgrid(self.x_Vector, self.y_Vector)  # 建立x,y坐标网格

    def ModifyParams(self, xgap=100, ygap=100, xextent=None, yextent=None, k=1):
        self.xgap = xgap
        self.ygap = ygap
        self.xextent = xextent
        self.yextent = yextent
        self.k = k
        self.x_Vector = np.linspace(-self.xextent, self.xextent, self.xgap)
        self.y_Vector = np.linspace(-self.yextent, self.yextent, self.ygap)  # 在范围（-10,10）之间，以x_gap，y_gap采样
        self.r_xGrid, self.r_yGrid = np.meshgrid(self.x_Vector, self.y_Vector)  # 建立x,y坐标网格

    def Wave(self):
        if self.method == 'Plane':
            Delta = self.k[0] * self.r_xGrid + self.k[1] * self.r_yGrid - (
                        self.k[0] ^ 2 + self.k[1] ^ 2) * self.c * self.t  # 等时相位
            E = self.A * np.cos(Delta)  # 光场分布
        elif self.method == 'Spheral':
            Delta = self.k * (self.r_xGrid ** 2 + self.r_yGrid ** 2) - self.k * self.c * self.t  # 等时相位
            E = self.A * np.cos(Delta) / np.sqrt(self.r_xGrid ** 2 + self.r_yGrid ** 2)  # 光场分布
        elif self.method == 'Interference':
            E = self.A * np.cos(np.cos(self.k[0] * self.r_xGrid)) * np.cos(np.sin(self.k[0] * self.r_xGrid))  # 光场分布
        else:
            E = 0
        return E

    def plotshow(self, E):
        fig, ax = plt.subplots()  # 建立画布和轴
        cont = plt.contourf(self.r_xGrid, self.r_yGrid, E, cmap=cm.gray)  # 建立等相面赋予对象cont
        fig.canvas.manager.set_window_title('%s' % self.method)  # 画布对象标题
        cont_colorbar = plt.colorbar(cont)  # 做cont对象的颜色条
        ax.set_title('%sEqualPhase' % self.method)  # 轴对象设置标题
        ax.set_xlabel('xDirection')  # 轴对象坐标标识
        ax.set_ylabel('yDirection')
        cont_colorbar.set_label('Amplitude')
        plt.show()  # 显示图像结果


if __name__ == '__main__':
    # 建立基础仿真对象
    OptBasic = OpticalBasicSimulation(A=1, c=1, t=1, k=None, xextent=10, yextent=10)
    # 平面波仿真
    OptBasic.method = 'Plane'
    PlaneE = OptBasic.Wave()
    OptBasic.plotshow(PlaneE)
    # 干涉仿真
    OptBasic.method = 'Interference'
    InterferenceE = OptBasic.Wave()
    OptBasic.plotshow(InterferenceE)
    # 球面波仿真
    OptBasic.method = 'Spheral'
    OptBasic.ModifyParams(k=20 * np.pi, xextent=0.5, yextent=0.5)
    SpheralE = OptBasic.Wave()
    OptBasic.plotshow(SpheralE)
