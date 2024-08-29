import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import math
import os
import sys
import threading
import re



######################定义若干函数#########################


# 读取txt文件中的矩形数据，返回矩形列表和图片尺寸
def read_rectangles_from_txt(file_path):
    """
    读取txt文件中的矩形数据，返回矩形列表和图片尺寸
    :param file_path: txt文件路径
    :return: 矩形列表和图片尺寸
    """
    rectangles = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines[:-1]:  #从第一行到倒数第二行
            x, y, length, width = map(float, line.split())
            x1 = x - length / 2
            y1 = y - width / 2
            x2 = x + length / 2
            y2 = y + width / 2
            rectangles.append(((x1, y1), (x2, y2)))
        # 解析图片尺寸,在最后一行
        match = re.match(r'\((\d+), (\d+)\)', lines[-1])
        if match:
            image_size = tuple(map(int, reversed(match.groups())))
        else:
            raise ValueError("Invalid image size format in txt file.")
    return rectangles, image_size


#定义初始化全局变量函数
def init_global_var(all_img_flag_in=False,file_path_in=None,save_path_in=None,count=1):
    """
    初始化全局变量
    :param all_flage: 是否全部标注
    :param file_path: txt文件路径,如果all_flage为False,则需要提供文件路径,否则不需要
    :param save_path: 保存路径
    :param count: 初始化全局变量的次数
    :return: None
    """
    global numbers_of_col,numbers_of_row,col_slope_error_threshold,max_intersection_area_ratio  #定义全局变量，并初始化
    global center_xy,center_to_rectangle,rectangle_to_center,col_k_list,col_b_list,row_k_list,row_b_list,col_k_list_sorted,col_b_list_sorted  #定义全局变量，并初始化
    global row_k_list_by_regression,row_b_list_by_regression,points_for_col_regression,coordinates_for_col_regression,points_for_row_regression  #定义全局变量，并初始化
    global points_to_matrix_dict,points_for_row_regression_by_rect,coordinates_for_row_regression_by_rect,Matrix  #定义全局变量，并初始化
    global nums_of_col,nums_of_row  #定义全局变量，并初始化
    global coordinates,img_size,img_name,save_dir  #定义全局变量，并初始化
    global all_img_flag,file_path,save_path,img_path,save_flag  #定义全局变量，并初始化
    #设定行数(暂时只针对6*5的行列进行拟合)
    nums_of_row=6
    #设定列数(暂时只针对6*5的行列进行拟合)
    nums_of_col=5
    numbers_of_col=5#列数
    numbers_of_row=6#行数
    #确定最大相交面积占比阈值
    max_intersection_area_ratio=0.01467314308844363
    #判定每列前一二个点是否拟合正确的斜率相对误差阈值（百分比）
    col_slope_error_threshold= 69.0661
    #创建存储中心坐标的空列表
    center_xy=[]
    # 创建一个字典来存储中心坐标与矩形坐标对的对应关系
    center_to_rectangle = {}
    #创建一个字典来存储矩形坐标对与中心坐标的对应关系
    rectangle_to_center = {}
    #创建一个列表来存储列直线的斜率
    col_k_list=[]
    #创建一个列表来存储行直线的截距
    col_b_list=[]
    #创建一个列表来存储行直线的斜率
    row_k_list=[]
    #创建一个列表来存储行直线的截距
    row_b_list=[]
    col_k_list_sorted=[]  # 用于存储列直线斜率排序后的列表
    col_b_list_sorted=[]  # 用于存储列直线截距排序后的列表
    #创建一个列表来存储拟合行直线的斜率
    row_k_list_by_regression=[]
    #创建一个列表来存储拟合行直线的截距
    row_b_list_by_regression=[]
    points_for_col_regression = []  # 用于存储列直线拟合点坐标列表
    coordinates_for_col_regression = []  # 用于存储列直线拟合矩形坐标列表
    points_for_row_regression=[]  # 用于存储行直线拟合点坐标列表
    Matrix=[] #用于存储行列对应坐标矩阵，Matrix[i][j]表示第i+1行第j+1列的中心坐标
    points_to_matrix_dict={} #用于存储点坐标到矩阵坐标的映射字典,points_to_matrix_dict[M[i][j]]表示第i+1行第j+1列点的行列坐标
    points_for_row_regression_by_rect=[]  # 用于存储行直线拟合矩形坐标列表
    coordinates_for_row_regression_by_rect=[]  # 用于存储行直线拟合矩形坐标列表
    all_img_flag=all_img_flag_in
    save_flag=all_img_flag
    file_path=file_path_in
    save_path=save_path_in
    if count==1:
        if all_img_flag:    #读取整个文件夹下所有图片进行行列拟合
            if len(sys.argv)!= 2:
                print("用法: python regression.py <image_path>")   #读取命令行参数
                sys.exit(1)
            img_path = sys.argv[1]     # 读取图像路径
            def redirect_output_to_files(normal_file, error_file):    # 定义一个函数，将终端打印的信息追加到指定的文件中
                sys.stdout = open(normal_file, 'a')  # 以追加模式打开普通输出文件
                sys.stderr = open(error_file, 'a')   # 以追加模式打开错误输出文件
            normal_file = save_path + "/normal_output.txt"
            error_file = save_path + "/error_output.txt"
            redirect_output_to_files(normal_file, error_file)  # 调用函数，将输出追加到文件中
        else:#读取单张图片进行行列拟合
            img_path=file_path   # 读取图像路径
                #读取矩形框坐标对列表和图像大小
    img_name= os.path.splitext(os.path.basename(img_path))[0]  # 获取图像名
    coordinates,img_size=read_rectangles_from_txt(img_path) # 存储矩形框坐标对列表和图像大小
    # 创建绘图对象和子图，设置图像大小
    
    save_dir=save_path  # 保存路径
    if not os.path.exists(save_dir):  # 如果保存路径不存在，则创建保存路径
        os.makedirs(save_dir)
    if count==1:
        global fig,ax
        fig, ax = plt.subplots(figsize=(img_size[0]/100, img_size[1]/100)) 
        print(img_name)  #打印图像名
        print("img_size:",img_size)  #打印图像大小


#从points列表中删除plots列表中的点，返回剩余的点坐标构成的列表
def remove_plots_from_points(points, plots):
    """
    从points列表中删除plots列表中的点，返回剩余的点坐标构成的列表
    :param points: 点坐标列表
    :param plots: 要删除的点坐标列表
    :return: 剩余的点坐标构成的列表
    """
    for plot in plots:   # 遍历 plots 列表中的每个元素
        try:  # 尝试从 points 列表中移除当前 plot
            points.remove(plot)
        except ValueError:  # 如果 plot 不在 points 列表中，忽略该异常
            pass
    return points


#计算与给定点距离最近的某点,并返回该点的坐标(利用中心坐标)（注意points是坐标列表）
def find_nearest_point(point, points):
    """
    计算与给定点距离最近的某点,并返回该点的坐标(利用中心坐标)（注意points是坐标列表）
    :param point: 给定点坐标
    :param points: 点坐标列表
    :return: 距离给定点最近点的坐标  
    """
    min_distance = float('inf')  # 设置一个初始的最小距离，设为无穷大
    nearest_point = None  # 初始化最近点的坐标为None
    for p in points:   # 遍历 points 列表中的每个元素
        distance = math.sqrt((p[0] - point[0])**2 + (p[1] - point[1])**2)  # 计算点到给定点的距离
        if (distance < min_distance) and (distance!= 0):  # 如果当前点的距离比最小距离小，则更新最小距离和最近点的坐标
            min_distance = distance
            nearest_point = p
    return nearest_point # 返回距离给定点最近点的坐标


#找到中心坐标中第n大的y坐标对应的点坐标
def find_nth_largest_y_coordinate(points, n):
    """
    找到中心坐标中第n大的y坐标对应的点坐标
    :param points: 点坐标列表
    :param n: 第n大
    :return: 第n大y坐标对应的点坐标
    """
    if len(points) < n:  # 如果点的数量少于n个，则无法找到第n大的y坐标
        return None
    unique_y_coordinates = set()   # 使用一个集合来存储y坐标，以便找到第n大的y值
    for point in points:
        unique_y_coordinates.add(point[1])
    sorted_y_coordinates = sorted(unique_y_coordinates, reverse=True)     # 将y坐标排序，并找到第n大的值
    nth_largest_y = sorted_y_coordinates[n - 1]
    for point in points: # 找到第n大的y值对应的x坐标
        if point[1] == nth_largest_y:
            return point


#从给定的矩形中心坐标中找到矩形右下角顶点坐标y值最大的点坐标，并返回该顶点对应的中心坐标
def find_nth_largest_y_coordinate_vertex(points, n,positoin=0):
    """
    从给定的矩形中心坐标中找到矩形右下角顶点坐标y值最大的点坐标，并返回该顶点对应的中心坐标
    :param points: 点坐标列表
    :param n: 第n大
    :param positoin: 0表示取最大值，1表示取右下角顶点，-1表示取左上角顶点
    :return: 第n大y坐标对应的点坐标
    """
    if len(points) < n:    # 如果点的数量少于n个，则无法找到第n大的y坐标
        return None
    unique_rect_coordinates = set()  # 使用一个集合来存储矩形坐标对，以便找到第n大的y值
    for point in points:
        unique_rect_coordinates.add(center_to_rectangle[point])
    sorted_rect_coordinates = sorted(unique_rect_coordinates, key=lambda x: x[1][1],reverse=True)  # 将矩形坐标对按右下角坐标的y值排序（从小到大），并找到第n大的值
    #如果最大值有重复，则取对应的x更大（position=1时）或更小（position=-1时）的点
    if sorted_rect_coordinates[n - 1][1][1] == sorted_rect_coordinates[n-2][1][1]:
        if positoin==1:
            if sorted_rect_coordinates[n - 1][1][0] > sorted_rect_coordinates[n-2][1][0]:
                y_max_point=rectangle_to_center[sorted_rect_coordinates[n - 1]]
            else:
                y_max_point=rectangle_to_center[sorted_rect_coordinates[n-2]]
        if positoin==-1:
            if sorted_rect_coordinates[n - 1][1][0] < sorted_rect_coordinates[n-2][1][0]:
                y_max_point=rectangle_to_center[sorted_rect_coordinates[n - 1]] 
            else:
                y_max_point=rectangle_to_center[sorted_rect_coordinates[n-2]]
        else:
            y_max_point=rectangle_to_center[sorted_rect_coordinates[n - 1]]
    else:
        y_max_point=rectangle_to_center[sorted_rect_coordinates[n - 1]]
    return y_max_point


#依据给定的摄像头位置position,寻找距离给定中心坐标对应的矩形顶点最近的矩形顶点，进而找到用于拟合的下一个中心坐标
#position参数用来指定搜索的范围（position=-1表示搜索x坐标小于给定点的点，position=1表示搜索x坐标大于给定点的点，position=0表示搜索所有点）
def find_nearest_point_by_rectangulat_vertex(point, points,position=0):
    """
    依据给定的摄像头位置position,寻找距离给定中心坐标对应的矩形顶点最近的矩形顶点，进而找到用于拟合的下一个中心坐标
    :param point: 给定点坐标
    :param points: 点坐标列表
    :param position: -1表示搜索x坐标小于给定点的点，1表示搜索x坐标大于给定点的点，0表示搜索所有点
    :return: 距离给定点最近点的坐标
    """
    min_distance = float('inf')  # 设置一个初始的最小距离，设为无穷大
    nearest_point = point  # 初始化最近点的坐标为point
    filtered_points=[]
    # 根据 position 参数的值来决定搜索空间
    if position == -1:  # 只考虑 x 坐标小于给定点的情况
        rectangle_vertex_now=(center_to_rectangle[point][0][0],center_to_rectangle[point][0][1]) #左侧时获取给定点的矩形的左上角坐标
        rectangles_coordiantes=[] #存储矩形坐标对
        for p in points:
            if p[0] < point[0]:
                filtered_points.append(p) #只搜寻小于给定点x坐标的点
                rectangles_coordiantes.append(center_to_rectangle[p]) #存储对应搜寻点的矩形坐标对
        for rectangle_vertex in rectangles_coordiantes:         # 遍历所有矩形右下角顶点坐标找到距离最近的矩形
            rectangle_vertex_right_bottom=(rectangle_vertex[1][0],rectangle_vertex[1][1])
            distance = math.sqrt((rectangle_vertex_right_bottom[0] - rectangle_vertex_now[0])**2 + (rectangle_vertex_right_bottom[1] - rectangle_vertex_now[1])**2)
            if distance < min_distance and distance != 0:
                min_distance = distance
                nearest_point = rectangle_to_center[rectangle_vertex] #更新最近点的坐标为距离该点最近的矩形的中心坐标
    elif position == 1:  # 只考虑 x 坐标大于给定点的情况
        rectangle_vertex_now=(center_to_rectangle[point][1][0],center_to_rectangle[point][0][1]) #右侧时获取给定点的矩形的右上角坐标
        rectangles_coordiantes=[] #存储矩形坐标对
        for p in points:
            if p[0] > point[0]:
                filtered_points.append(p) #只搜寻大于给定点x坐标的点
                rectangles_coordiantes.append(center_to_rectangle[p]) #存储对应搜寻点的矩形坐标对
        for rectangle_vertex in rectangles_coordiantes:      # 遍历所有矩形左下角顶点坐标找到距离最近的矩形
            rectangle_vertex_left_bottom=(rectangle_vertex[0][0],rectangle_vertex[0][1])
            distance = math.sqrt((rectangle_vertex_left_bottom[0] - rectangle_vertex_now[0])**2 + (rectangle_vertex_left_bottom[1] - rectangle_vertex_now[1])**2)
            if distance < min_distance and distance != 0:
                min_distance = distance
                nearest_point = rectangle_to_center[rectangle_vertex] #更新最近点的坐标为距离该点最近的矩形的中心坐标
    else:     # 考虑所有点
        filtered_points = points # 存储所有点坐标
        for p in points:  # 遍历所有中心坐标找到距离最近的矩形
            distance = math.sqrt((p[0] - point[0])**2 + (p[1] - point[1])**2)
            if distance < min_distance and distance != 0:
                min_distance = distance
                nearest_point = p #更新最近点的坐标为距离该点最近的中心坐标
    #print("最近点坐标：",nearest_point)
    return nearest_point


#对points里的坐标进行直线拟合，返回斜率和截距(注意points要是numpy数组)
def linear_regression(points):
    """
    对points里的坐标进行直线拟合，返回斜率和截距(注意points要是numpy数组)
    :param points: 点坐标列表
    :return: 斜率和截距
    """
    [vx, vy, x, y] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)  # 调用opencv的fitLine函数进行直线拟合
    # 计算斜率和截距
    k= vy / vx
    b = y - k * x
    return k, b # 返回斜率和截距


#判断直线（y=kx+b）是否与矩形（左上角坐标为p[0]，右下角坐标为p[1]）相交，返回True或False
def is_line_intersect_rectangle(k, b, p):
    """
    判断直线（y=kx+b）是否与矩形（左上角坐标为p[0]，右下角坐标为p[1]）相交，返回True或False
    :param k: 斜率
    :param b: 截距
    :param p: 矩形坐标对
    :return: True或False
    """
    # 提取矩形坐标
    if k==0:
        col_flag=False #是否为列直线
        #print("斜率为0，为列直线")
        draw_center_point_and_rect(center_xy,ax,img_size) #绘制中心点和矩形框
        set_axis_and_save(col_flag,ax,img_name,img_size,save_dir,save_flag)#设置坐标轴范围和保存路径等
        exit() #退出程序
    assert k!= 0, "斜率不能为0"
    rect_x1, rect_y1 = p[0][0], p[0][1]  # 左上角坐标
    rect_x2, rect_y2 = p[1][0], p[1][1]  # 右下角坐标
    # 检查左边界 x = rect_x1
    y_left = k * rect_x1 + b
    if rect_y1 <= y_left <= rect_y2:
        return True
    # 检查右边界 x = rect_x2
    y_right = k * rect_x2 + b
    if rect_y1 <= y_right <= rect_y2:
        return True
    # 检查上边界 y = rect_y2
    x_top = (rect_y2 - b) / k
    if rect_x1 <= x_top <= rect_x2:
        return True
    # 检查下边界 y = rect_y1
    x_bottom = (rect_y1 - b) / k
    if rect_x1 <= x_bottom <= rect_x2:
        return True
    return False


#定义绘制直线函数
def draw_line(k,b,ax,c,begin_x,end_x):
    """
    定义绘制直线函数
    :param k: 斜率
    :param b: 截距
    :param ax: 绘图对象
    :param c: 颜色
    :param begin_x: 直线起点x坐标
    :param end_x: 直线终点x坐标
    :return: None
    """
    x = np.array([begin_x, end_x])  # 定义x坐标
    y = k * x + b  # 计算y坐标
    ax.plot(x, y, color=c, linewidth=2)    # 绘制直线


#拟合列直线（距离摄像头较近位置，如一二列）
def loop_col_regression(first_point,points,ax):
    """
    拟合列直线（距离摄像头较近位置，如一二列）
    :param first_point: 第一个点坐标
    :param points: 点坐标列表
    :param ax: 绘图对象
    :rertun: 斜率和截距
    """
    #print("第1个点坐标：",first_point)
    points_for_regression = []#定义一个空列表,用于存储用于拟合的点
    rectangles_for_regression = [] #定义一个空列表,用于存储用于拟合的矩形框坐标对
    #存入拟合起始点坐标
    points_for_regression.append(tuple(first_point))
    #存入拟合起始矩形坐标对
    rectangles_for_regression.append(center_to_rectangle[tuple(first_point)])
    #center_xy_copy = list(remove_plots_from_points(points,first_point)) #从中心坐标列表中删除拟合过的坐标
    # 找到与第一个中心点距离最近的点
    point_now = find_nearest_point(first_point, remove_plots_from_points(points,points_for_regression))
    #center_xy_copy = list(remove_plots_from_points(points,point_now)) #从中心坐标列表中删除拟合过的坐标
    #print("第2个点坐标：",point_now)
    #存入当前的拟合点坐标
    points_for_regression.append(point_now)
    #存入当前的拟合矩形坐标对
    rectangles_for_regression.append(center_to_rectangle[point_now])
    i=1
    #print("第",i,"次拟合完成")
    while i<nums_of_row:
        #对存好的点进行直线拟合
        k, b = linear_regression(np.array(points_for_regression))
        #找到与当前点最近的点的坐标
        point_next = find_nearest_point(point_now, remove_plots_from_points(points,points_for_regression))
        #print("第",i,"次拟合点坐标：",point_next)
        #判断拟合的直线与下一个矩形是否相交
        if is_line_intersect_rectangle(k, b, center_to_rectangle[point_next]):
            #相交，则将该点坐标加入到拟合点中
            points_for_regression.append(point_next)
            #存入当前的拟合矩形坐标对
            rectangles_for_regression.append(center_to_rectangle[point_next])
            #center_xy_copy = list(remove_plots_from_points(points,point_next)) #从中心坐标列表中删除拟合过的坐标
            #继续进行直线拟合
            i+=1
            #更新当前点坐标
            point_now = point_next
        #    print("第",i+2,"个点坐标：",point_now) 
        #    print("第",i,"次拟合完成")
        else:
            #不相交,该列拟合完成
            #print("不相交")
            break
    #draw_line(k,b,ax,'green',0,img_size[0]) #绘制拟合的直线
    col_k_list.append(k) #记录斜率
    col_b_list.append(b) #记录截距
    #points=list(remove_plots_from_points(points,points_for_regression)) #从中心坐标列表中删除拟合过的坐标
    #points=list(remove_plots_from_points(points,[first_point])) #从中心坐标列表中删除第一个坐标
    return points_for_regression,rectangles_for_regression #返回拟合的点坐标和拟合的矩形坐标对


#找出剩余的矩形坐标对
def function_coordinates_rectangles_remain(center_xy_copy):
    """
    找出剩余的矩形坐标对
    :param center_xy_copy: 剩余的中心坐标列表
    :return: 剩余的矩形坐标对列表
    """
    coordinates_copy = []#创建一个空列表，用于存储剩余的矩形坐标对
    for plot in center_xy_copy:
        coordinates_copy.append(center_to_rectangle[plot])
    return coordinates_copy


#判断给定斜率和截距的直线y=kx+b是否与任意个给定的矩形框coordinates（列表，元素为元组坐标对（包含左上角坐标(x1,y1)和右下角坐标(x2,y2)））相交，
#如果相交，返回True和相交矩形的中心坐标列表，否则返回False。
#position表示摄像头位置，-1表示左侧，1表示右侧，0表示不限制位置(-1表示只从起点的左侧寻找矩形，1表示只从起点的右侧寻找矩形，0表示不限制位置)
#begin_point表示直线的起点坐标
def line_rect_intersection(k,b,coordinates,begin_point,position):
    """
    判断给定斜率和截距的直线y=kx+b是否与任意个给定的矩形框coordinates（列表，元素为元组坐标对（包含左上角坐标(x1,y1)和右下角坐标(x2,y2)））相交，
    如果相交，返回True和相交矩形的中心坐标列表，否则返回False。
    :param k: 斜率
    :param b: 截距
    :param coordinates: 矩形坐标对列表
    :param begin_point: 直线的起点坐标
    :param position: -1表示左侧，1表示右侧，0表示不限制位置(-1表示只从起点的左侧寻找矩形，1表示只从起点的右侧寻找矩形，0表示不限制位置)
    :return: True和相交矩形的中心坐标列表，否则返回False。
    """
    flag=False
    center_point=[] #相交矩形的中心坐标
    if k==0:
        col_flag=False #是否为列直线
        draw_center_point_and_rect(center_xy,ax,img_size) #绘制中心点和矩形框
        set_axis_and_save(col_flag,ax,img_name,img_size,save_dir,save_flag)#设置坐标轴范围和保存路径等
        exit() #退出程序
    assert k!= 0, "斜率不能为0"
    for plot in coordinates :
        center_xy_plot=(plot[0][0]+plot[1][0])/2, (plot[0][1]+plot[1][1])/2 #矩形中心坐标
        if position==-1:
            if center_xy_plot[0]<begin_point[0] :
                (x1,y1),(x2,y2) = plot #提取矩形坐标
                #检查左边界
                x=x1
                y_left=k*x+b
                if y1<=y_left<=y2:
                    flag=True 
                    center_point.append(center_xy_plot)  #直线与矩形相交
                    continue
                #检查右边界
                x=x2
                y_right=k*x+b
                if y1<=y_right<=y2:
                    flag=True
                    center_point.append(center_xy_plot)  #直线与矩形相交
                    continue
                #检查上边界
                y=y1
                x_up=(-b-y)/k
                if x1<=x_up<=x2:
                    flag=True
                    center_point.append(center_xy_plot)  #直线与矩形相交
                    continue
                #检查下边界
                y=y2
                x_down=(-b-y)/k
                if x1<=x_down<=x2:
                    flag=True
                    center_point.append(center_xy_plot)  #直线与矩形相交
                    continue
                #检查上边界
                y=y1
                x_top=(y-b)/k
                if x1<=x_top<=x2:
                    flag=True
                    center_point.append(center_xy_plot)  #直线与矩形相交
                    continue
                #检查下边界
                y=y2
                x_bottom=(y-b)/k
                if x1<=x_bottom<=x2:
                    flag=True      
                    center_point.append(center_xy_plot)  #直线与矩形相交
                    continue
            else:
                continue
        elif position==1:
            if center_xy_plot[0]>begin_point[0] :
                (x1,y1),(x2,y2) = plot #提取矩形坐标
                
                #检查左边界
                x=x1
                y_left=k*x+b
                if y1<=y_left<=y2:
                    flag=True 
                    center_point.append(center_xy_plot)  #直线与矩形相交
                    continue
                #检查右边界
                x=x2
                y_right=k*x+b
                if y1<=y_right<=y2:
                    flag=True
                    center_point.append(center_xy_plot)  #直线与矩形相交
                    continue
                #检查上边界
                y=y1
                x_up=(-b-y)/k
                if x1<=x_up<=x2:
                    flag=True
                    center_point.append(center_xy_plot)  #直线与矩形相交
                    continue
                #检查下边界
                y=y2
                x_down=(-b-y)/k
                if x1<=x_down<=x2:
                    flag=True
                    center_point.append(center_xy_plot)  #直线与矩形相交
                    continue
                #检查上边界
                y=y1
                x_top=(y-b)/k
                if x1<=x_top<=x2:
                    flag=True
                    center_point.append(center_xy_plot)  #直线与矩形相交
                    continue
                #检查下边界
                y=y2
                x_bottom=(y-b)/k
                if x1<=x_bottom<=x2:
                    flag=True      
                    center_point.append(center_xy_plot)  #直线与矩形相交
                    continue
            else:
                continue
        else:
            if begin_point!=center_xy_plot: #排除起点坐标
                (x1,y1),(x2,y2) = plot #提取矩形坐标
                
                #检查左边界
                x=x1
                y_left=k*x+b
                if y1<=y_left<=y2:
                    flag=True 
                    center_point.append(center_xy_plot)  #直线与矩形相交
                    continue
                #检查右边界
                x=x2
                y_right=k*x+b
                if y1<=y_right<=y2:
                    flag=True
                    center_point.append(center_xy_plot)  #直线与矩形相交
                    continue
                #检查上边界
                y=y1
                x_up=(-b-y)/k
                if x1<=x_up<=x2:
                    flag=True
                    center_point.append(center_xy_plot)  #直线与矩形相交
                    continue
                #检查下边界
                y=y2
                x_down=(-b-y)/k
                if x1<=x_down<=x2:
                    flag=True
                    center_point.append(center_xy_plot)  #直线与矩形相交
                    continue
                #检查上边界
                y=y1
                x_top=(y-b)/k
                if x1<=x_top<=x2:
                    flag=True
                    center_point.append(center_xy_plot)  #直线与矩形相交
                    continue
                #检查下边界
                y=y2
                x_bottom=(y-b)/k
                if x1<=x_bottom<=x2:
                    flag=True      
                    center_point.append(center_xy_plot)   #直线与矩形相交
                    continue
            else:#直线与矩形框不相交
                continue                        
    return flag,center_point    


#判断给定斜率和截距的直线y=kx+b是否与任意个给定的矩形框coordinates（列表，元素为元组坐标对（包含左上角坐标(x1,y1)和右下角坐标(x2,y2)））相交，
#如果相交，返回True和相交矩形的中心坐标列表，否则返回False。
#当前正在使用
def check_line_intersects_rectangles(k, b, coordinates):
    """
    判断给定斜率和截距的直线y=kx+b是否与任意个给定的矩形框coordinates（列表，元素为元组坐标对（包含左上角坐标(x1,y1)和右下角坐标(x2,y2)））相交，
    如果相交，返回True和相交矩形的中心坐标列表，否则返回False。
    :param k: 斜率
    :param b: 截距
    :param coordinates: 矩形坐标对列表
    :return: True和相交矩形的中心坐标列表，否则返回False
    """
    def does_line_intersect_rect(k, b, x1, y1, x2, y2):
        if x1 > x2: x1, x2 = x2, x1
        if y1 < y2: y1, y2 = y2, y1
        # 左边界 (x1, y)
        y_left = k * x1 + b
        if y2 <= y_left <= y1:
            return True
        # 右边界 (x2, y)
        y_right = k * x2 + b
        if y2 <= y_right <= y1:
            return True
        # 上边界 (x, y1)
        if k != 0:
            x_top = (y1 - b) / k
            if x1 <= x_top <= x2:
                return True
        # 下边界 (x, y2)
        if k != 0:
            x_bottom = (y2 - b) / k
            if x1 <= x_bottom <= x2:
                return True
        return False
    #计算矩形中心坐标
    def get_center_of_rect(x1, y1, x2, y2):
        if x1 > x2: x1, x2 = x2, x1
        if y1 < y2: y1, y2 = y2, y1
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    intersecting_centers = []
    for rect in coordinates:
        (x1, y1), (x2, y2) = rect
        if does_line_intersect_rect(k, b, x1, y1, x2, y2):
            intersecting_centers.append(get_center_of_rect(x1, y1, x2, y2))
    if intersecting_centers:
        return True, intersecting_centers
    else:
        return False, []


#拟合列直线(best) 
def loop_col_regression_best(first_point,points,ax,position): 
    """
    拟合列直线(best) 
    :param first_point: 第一个点坐标
    :param points: 点坐标列表
    :param ax: 绘图对象
    :param position: -1表示左侧，1表示右侧，0表示不限制位置(-1表示只从起点的左侧寻找矩形，1表示只从起点的右侧寻找矩形，0表示不限制位置)
    :return: 斜率和截距
    """
    #print("第1个点坐标：",first_point)
    #定义一个空列表,用于存储用于拟合的点
    points_for_regression = []
    #定义一个空列表，用于存储用于拟合的矩形坐标对
    rectangles_for_regression = []
    #存入拟合起始点坐标
    points_for_regression.append(tuple(first_point))
    #存入拟合起始矩形坐标对
    rectangles_for_regression.append(center_to_rectangle[tuple(first_point)])
    # 找到与第一个中心点距离最近的点
    point_now = find_nearest_point_by_rectangulat_vertex(tuple(first_point), remove_plots_from_points(points,points_for_regression),position)
    #print("可能的第2个点坐标：",point_now)
    error_2rd_point1=point_now
    k, b = linear_regression(np.array([first_point,point_now]))#对第一个点和第二个点进行直线拟合
    k_flag=True
    if len(col_k_list)>1:    
        col_k_avg=sum(col_k_list[1:len(col_k_list)])/(len(col_k_list)-1)
        k_relative_error=abs((k-col_k_avg)/col_k_avg)*100
        #print("斜率相对误差：",k_relative_error,"%")
        if k_relative_error>col_slope_error_threshold:
            k_flag=False
    is_line_intersect_rectangle_flag=0
    if k_flag:
        for p in center_xy:
            if p!=tuple(first_point) and p!=error_2rd_point1 :
                if is_line_intersect_rectangle(k, b, center_to_rectangle[p])==1:
                    #print("第一个点与第二个点的斜率与前列斜率的相对误差小于阈值，并且至少穿过一个矩形，说明第二个点寻找正确")
                    points_for_regression.append(point_now)
                    points=remove_plots_from_points(points,points_for_regression)
                    #print("第",2,"个点坐标：",point_now)
                    rectangles_for_regression.append(center_to_rectangle[point_now])
                    is_line_intersect_rectangle_flag=1
                    break
    if is_line_intersect_rectangle_flag==0 or k_flag==False:    
        if k_flag: print("不穿过任何矩形,重新寻找")
        if k_flag==False:
            print("不穿过任何矩形，而且斜率相对误差大于阈值，重新寻找")
        points_for_regression_cp=[tuple(first_point),point_now]
        #print(points_for_regression_cp)
        points_copy=points.copy()
        points_copy=list(remove_plots_from_points(points_copy,points_for_regression_cp))
        #print(points_copy)
        point_now=find_nearest_point_by_rectangulat_vertex(tuple(first_point), points_copy,position)
        #print("重新寻找的第2个点坐标：",point_now)
        error_2rd_point2=point_now
        k,b = linear_regression(np.array([first_point,point_now]))#对第一个点和第二个点进行直线拟合
        k_flag=True
        if len(col_k_list)>1:    
            col_k_avg=sum(col_k_list[1:len(col_k_list)])/(len(col_k_list)-1)
            k_relative_error=abs((k-col_k_avg)/col_k_avg)*100
            #print("斜率相对误差：",k_relative_error,"%")
            if k_relative_error>col_slope_error_threshold:
                points_for_regression_cp.append(error_2rd_point2)
                k_flag=False
        is_line_intersect_rectangle_flag=0
        for p in center_xy:
            if p!=tuple(first_point) and p!=error_2rd_point1 and p!=error_2rd_point2:
                if is_line_intersect_rectangle(k, b, center_to_rectangle[p])==1 and k_flag:
                    #print("第一个点与第二个点的斜率与前列斜率的相对误差小于阈值，并且至少穿过一个矩形，说明第二个点寻找正确")
                    #points_for_regression.append(error_2rd_point2)
                    #points=remove_plots_from_points(points,points_for_regression)
                    #print("第",2,"个点坐标：",error_2rd_point2)
                    #rectangles_for_regression.append(center_to_rectangle[error_2rd_point2])
                    is_line_intersect_rectangle_flag=1
                    break
        if is_line_intersect_rectangle_flag==0 or k_flag==False:    
            if k_flag: print("不穿过任何矩形,重新寻找")
            if k_flag==False:
                print("不穿过任何矩形，而且斜率相对误差大于阈值，重新寻找")
            #print(points_for_regression_cp)
            points_copy=points.copy()
            points_copy=list(remove_plots_from_points(points_copy,points_for_regression_cp))
            #print(points_copy)
            point_now=find_nearest_point_by_rectangulat_vertex(tuple(first_point), points_copy,position)
            #print("重新寻找的第2个点坐标：",point_now)
        points_for_regression.append(point_now)
        rectangles_for_regression.append(center_to_rectangle[point_now])
        points=remove_plots_from_points(points,points_for_regression)
        #print(points_for_regression)
        #print(points)
        #存入当前的拟合点坐标
    #points_for_regression.append(point_now)
    #存入当前的拟合矩形坐标对
    #rectangles_for_regression.append(center_to_rectangle[point_now])
    i=1
    #print("第",i,"次拟合完成")
    while i<nums_of_row:
        #对存好的点进行直线拟合
        k, b = linear_regression(np.array(points_for_regression))
        #print("斜率：",k,"截距：",b)
        #找到与当前点最近的点的坐标
        point_next = find_nearest_point_by_rectangulat_vertex(point_now,remove_plots_from_points(points,points_for_regression),position)
        #print("第",i,"次拟合点坐标：",point_next)
        #判断拟合的直线与下一个矩形是否相交
        if is_line_intersect_rectangle(k, b, center_to_rectangle[point_next]):
            #print("第",i,"次拟合直线与最近点矩形相交")
            #继续进行直线拟合
            i+=1
            #更新当前点坐标
            if point_next!=point_now :#相交且两点不是同一点，则将该点坐标加入到拟合点中
                points_for_regression.append(point_next)
                rectangles_for_regression.append(center_to_rectangle[point_next])
                point_now = point_next
                #print("第",i+1,"个点坐标：",point_now) 
            #print("第",i,"次拟合完成")
        else:
            #print("point_now:",point_now)
            #print("points:",points)
            flag,center_point=line_rect_intersection(k, b, function_coordinates_rectangles_remain(points),point_now,position) #判断直线与矩形框是否相交,返回相交矩形的中心坐标列表
            if flag:
                #print("相交")
                #print("相交矩形的中心坐标为：",center_point)
                nearest_centerpoint=find_nearest_point_by_rectangulat_vertex(point_now,center_point,position) #找到距离直线起点最近的矩形中心坐标
                #print("第",i,"次拟合相交矩形的中心坐标为：",nearest_centerpoint)
                points_for_regression.append(nearest_centerpoint) #将距离最近的相交矩形的中心坐标加入到拟合点中
                rectangles_for_regression.append(center_to_rectangle[nearest_centerpoint]) #将距离最近的相交矩形的中心坐标对应的矩形坐标对加入到拟合矩形坐标对中
                continue
            else:
                if len(points_for_regression)==2:
                    #print("不相交且拟合点数小于等于3，重新寻找第二个点")
                    points_copy=points.copy()
                    points_for_regression_cp=points_for_regression.copy()
                    points_copy=list(remove_plots_from_points(points_copy,points_for_regression_cp))
                    #print(points_copy)
                    point_now=find_nearest_point_by_rectangulat_vertex(tuple(first_point), points_copy,position)
                    #print("重新寻找的第2个点坐标：",point_now)
                    points_for_regression.append(point_now)
                    rectangles_for_regression.append(center_to_rectangle[point_now])
                    points=remove_plots_from_points(points,points_for_regression)
                    #print(points_for_regression)
                    continue
                else:
                    #print("不相交")
                    break
    #draw_line(k,b,ax,'green',0,img_size[0]) #绘制拟合的直线
    col_k_list.append(k)#记录斜率
    col_b_list.append(b)#记录截距
    #print("绘制直线的斜率和截距：",k,b)
    return points_for_regression,rectangles_for_regression #返回拟合点坐标列表和拟合矩形坐标对列表


#创建一个矩阵Matrix，Matrix[i][j]表示第i+1行第j+1列的中心坐标
def construct_coordinate_matrix(points_for_row_regression, points_for_col_regression):
    """
    创建一个矩阵Matrix，Matrix[i][j]表示第i+1行第j+1列的中心坐标
    :param points_for_row_regression: 行方向的拟合点坐标列表
    :param points_for_col_regression: 列方向的拟合点坐标列表
    :return: 矩阵Matrix
    """
    num_rows = len(points_for_row_regression)
    num_cols = len(points_for_col_regression)
    Matrix = [[None] * num_cols for _ in range(num_rows)]  # 初始化矩阵
    for i in range(num_rows): # 填充矩阵
        for j in range(num_cols):
            Matrix[i][j] = points_for_row_regression[i][j]
    return Matrix #返回矩阵


#创建一个字典，将中心坐标映射到所在的行与列
def create_coordinate_dictionary(Matrix):
    """
    创建一个字典，将中心坐标映射到所在的行与列
    :param Matrix: 矩阵Matrix
    :return: 中心坐标-行列字典points_to_matrix_dict
    """
    points_to_matrix_dict = {} # 初始化字典
    cols=len(Matrix[0]) #列数
    rows=len(Matrix)  #行数
    for i in range(rows):  # 填充字典
        for j in range(cols):
            points_to_matrix_dict[Matrix[i][j]] = (i+1, j+1) 
    return points_to_matrix_dict  # 返回字典


#对输入一组坐标元组，按坐标的y值从大到小排序，
# 并输出排序后的坐标元组。
def sort_points_by_ymax_to_ymin(points):
    """
    对输入一组坐标元组，按坐标的y值从大到小排序，
    并输出排序后的坐标元组。
    :param points: 输入坐标元组
    :return: 排序后的坐标元组
    """
    # 按y值从大到小排序
    sorted_points = sorted(points, key=lambda x: x[1], reverse=True)
    return sorted_points


#读取整个文件夹内中图片进行拟合时用来关闭程序的函数
def close_plot():
    """
    读取整个文件夹内中图片进行拟合时用来关闭程序的函数
    """
    plt.close()
    sys.exit()


#去除列表中重复的元素
def remove_duplicates(lst):
    """
    去除列表中重复的元素
    :param lst: 输入列表
    :return: 去除重复元素后的列表
    """
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


#判断两个矩形是否有重叠部分，并计算相交面积占比
def rect_inter_ratio(rect1,rect2):
    """
    判断两个矩形是否有重叠部分，并计算相交面积占比
    :param rect1: 矩形1的坐标对
    :param rect2: 矩形2的坐标对
    :return: 1表示有重叠部分，返回0表示无重叠部分 和 返回相交面积占比
    """
    #提取矩形的坐标
    (x1_r1, y1_r1), (x2_r1, y2_r1) = rect1
    (x1_r2, y1_r2), (x2_r2, y2_r2) = rect2
    #检查是否有重叠部分
    if(max(x1_r1, x1_r2)<=min(x2_r1, x2_r2) and max(y1_r1, y1_r2)<=min(y2_r1, y2_r2)):#有重叠部分
        #计算相交面积
        intersection_x1=max(x1_r1, x1_r2)  #相交矩形的x1坐标
        intersection_y1=max(y1_r1, y1_r2)  #相交矩形的y1坐标
        intersection_x2=min(x2_r1, x2_r2)  #相交矩形的x2坐标
        intersection_y2=min(y2_r1, y2_r2)  #相交矩形的y2坐标
        #print("相交矩形的坐标为：",(intersection_x1, intersection_y1),(intersection_x2, intersection_y2))
        #intersection_patch = patches.Rectangle((intersection_x1, intersection_y1), intersection_x2-intersection_x1, intersection_y2-intersection_y1, linewidth=0, color='b')
        #ax.add_patch(intersection_patch)
        intersection_area = (intersection_x2-intersection_x1)*(intersection_y2-intersection_y1)
        #计算两个矩形的面积
        area1 = (x2_r1-x1_r1)*(y2_r1-y1_r1)
        area2 = (x2_r2-x1_r2)*(y2_r2-y1_r2)
        #计算相交面积占比
        intersection_area_ratio = intersection_area/(area1+area2-intersection_area)
        #print("相交面积占比：",intersection_area_ratio)
        intersection_area_ratio =intersection_area_ratio*(intersection_y2/img_size[1])  # 乘以相交部分矩形的y坐标/img_size[1]，使得相交面积占比与相交部分矩形的y坐标/img_size[1]成正比
        #print("相交面积占比(乘以了相交部分矩形的ymax/img_size[1]):", intersection_area_ratio)
        return 1,intersection_area_ratio
    else:
        return 0,0.0


#找出一列矩形中两两相交矩形面积的最大对应的那两个矩形
def find_intersect_max_area_rectangles(rectangles):
    """
    找出一列矩形中两两相交矩形面积的最大对应的那两个矩形
    :param rectangles: 一列矩形的坐标对列表
    :return: 1表示找到了相交矩形，返回0表示没有找到相交矩形 和 返回两个矩形的坐标max_rectangles[0],max_rectangles[1]，返回最大相交面积占比 max_area
    """
    max_area=0
    rect1=[]
    rect2=[]
    f=0
    for i in range(len(rectangles)):
        if i==len(rectangles)-1:
            break
        for j in range(i+1,len(rectangles)):
            if j==len(rectangles)-1:
                break
            f,s=rect_inter_ratio(rectangles[i],rectangles[j])
            if f and s>max_area:
                max_area=s
                rect1=rectangles[i]
                rect2=rectangles[j]
                f=1
    return f,rect1,rect2,max_area


#判断两个矩形是否有重叠部分，通过计算相交面积占比判断是否相交，并返回true or false，以及两相交矩形的合并矩形（目前经验值，缺乏进一步测试验证）
def merge_rectangles(rect1, rect2,max_intersection_area_ratio):
    """
    判断两个矩形是否有重叠部分，通过计算相交面积占比判断是否相交，并返回true or false，以及两相交矩形的合并矩形（目前经验值，缺乏进一步测试验证）
    :param rect1: 矩形1的坐标对
    :param rect2: 矩形2的坐标对
    :param max_intersection_area_ratio: 最大相交面积占比阈值
    :return: 1表示找到了相交矩形，返回0表示没有找到相交矩形，返回两相交矩形合并后的矩形，以及相交面积占比
    """
    #提取矩形的坐标
    (x1_r1, y1_r1), (x2_r1, y2_r1) = rect1
    (x1_r2, y1_r2), (x2_r2, y2_r2) = rect2
    flag,intersection_area_ratio=rect_inter_ratio(rect1,rect2)
    if flag :
        #有重叠部分
        if intersection_area_ratio>max_intersection_area_ratio:
            #print("相交面积占比大于阈值，为",intersection_area_ratio)
            #print("进行合并")
            #计算合并后的矩形
            center_x = (x1_r1+x2_r1+x1_r2+x2_r2)/4
            center_y = (y1_r1+y2_r1+y1_r2+y2_r2)/4
            width = ((x2_r1-x1_r1)+(x2_r2-x1_r2))/2
            height = ((y2_r1-y1_r1)+(y2_r2-y1_r2))/2
            merged_rectangle = ((center_x-width/2, center_y-height/2),(center_x+width/2, center_y+height/2))
            ax.text(merged_rectangle[0][0],merged_rectangle[1][1],str(round(intersection_area_ratio,4)),color='red',fontsize=10)
            #print("合并后的矩形坐标为：",merged_rectangle)
            #绘制合并后的矩形
            #merged_patch = patches.Rectangle(merged_rectangle[0], merged_rectangle[1][0]-merged_rectangle[0][0], merged_rectangle[1][1]-merged_rectangle[0][1], linewidth=5, edgecolor='green', facecolor='none')
            #ax.add_patch(merged_patch)
            return 1, merged_rectangle, intersection_area_ratio
        else:
            #print("相交面积占比小于阈值，为",intersection_area_ratio)
            #print("不进行合并")
            return 0, None,0.0
    else:
        #print("无重叠部分")
        return 0, None,0.0


#传入一个坐标point和该坐标所在的列表points，根据其对应的矩形坐标对的最左侧的x坐标，进行排序
def sort_points_by_rect_left_x(point, points):
    """
    传入一个坐标point和该坐标所在的列表points，根据其对应的矩形坐标对的最左侧的x坐标，进行排序，
    返回排序后的列表points
    :param point: 输入坐标
    :param points: 输入坐标所在的列表
    :return: 排序后的列表points
    """
    #根据矩形坐标对的最左侧的x坐标进行排序
    coordinates=[]
    left_up_points=[] #存储所有矩形左上角的点
    for p in points:
        coordinates.append(center_to_rectangle[p])
    for coordinate in coordinates:
        left_up_points.append(coordinate[0])
    #根据矩形左上角的x坐标进行排序(从小到大)
    left_up_points.sort(key=lambda x: x[0]) #从小到大排序
    now_left_up_point=center_to_rectangle[point][0]
    #print("排序后的点坐标：", sorted_points)
    for i in range(len(left_up_points)):
        if now_left_up_point == left_up_points[i]:
            order=i
            break
    return order+1


#传入一个坐标point和该坐标所在的列表points，根据其对应的矩形坐标对的最右侧的x坐标，进行排序，
def sort_points_by_rect_right_x(point, points):
    """
    传入一个坐标point和该坐标所在的列表points，根据其对应的矩形坐标对的最右侧的x坐标，进行排序，
    返回排序后的列表points
    :param point: 输入坐标
    :param points: 输入坐标所在的列表
    :return: 排序后的列表points
    """
    #根据矩形坐标对的最右侧的x坐标进行排序
    coordinates=[]
    left_up_points=[] #存储所有矩形右下角的点
    for p in points:
        coordinates.append(center_to_rectangle[p])
    for coordinate in coordinates:
        left_up_points.append(coordinate[1])
    #根据矩形右下角的x坐标进行排序(从大到小)
    left_up_points.sort(key=lambda x: x[0], reverse=True) #从大到小排序
    now_left_up_point=center_to_rectangle[point][1]
    for i in range(len(left_up_points)):
        if now_left_up_point == left_up_points[i]:
            order=i
            break
    return order+1


#判断找到的第一个点是否正确（也就是第一个点到底存不存在）
def is_first_point_correct(first_point, center_xy, position):
    """
    判断找到的第一个点是否正确（也就是第一个点到底存不存在）
    :param first_point: 找到的第一个点坐标
    :param center_xy: 所有矩形的中心坐标
    :param position: 摄像头位置
    :return: 1表示找到了正确的第一个点，返回0表示找到了错误的第一个点
    """
    #根据摄像头位置分类判断
    if position == -1:  # 摄像头在左侧
        order = sort_points_by_rect_left_x(first_point, center_xy)
        #print("第一个点的排序为zuo：", order)
        #plt.text(center_to_rectangle[first_point][0][0], center_to_rectangle[first_point][0][1], str(order), color='red', fontsize=10)
        if  1<=order<=9:  # 第一个点在第一列
            return 1  #说明找到了正确的第一个点
        else:
            return 0  #说明找到了错误的第一个点
    if position == 1:  # 摄像头在右侧
        order = sort_points_by_rect_right_x(first_point, center_xy)
        #print("第一个点的排序为right：", order)
        #plt.text(center_to_rectangle[first_point][1][0], center_to_rectangle[first_point][1][1], str(order), color='red', fontsize=10)
        if 1<=order<=9:  # 第一个点在第五列
            return 1  #说明找到了正确的第一个点
        else:
            return 0  #说明找到了错误的第一个点


#定义拟合全部列直线的函数
#传入第一个中心坐标first_point，中心坐标列表center_xy_copy，子图ax,摄像头位置position
#计算拟合的列直线的斜率k_list和截距b_list
#将每列的拟合点坐标到总的列表points_for_col_regression中
#将每列的拟合矩形坐标对列表添加到总的列表coordinates_for_col_regression中
#无返回值
def loop_all_col_regression(first_point, first_point_flag, center_xy, ax, position):
    """
    定义拟合全部列直线的函数
    :param first_point: 第一个中心坐标
    :param first_point_flag: 第一个中心坐标是否存在
    :param center_xy: 所有矩形的中心坐标
    :param ax: 子图ax
    :param position: 摄像头位置
    :return: 得到每列的拟合点坐标列表 points_for_col_regression
    """
    points_for_col_regression = []  # 用于存储每列的拟合点坐标列表
    #开始拟合列直线
    center_xy_copy = list(center_xy)  # 复制中心坐标列表
    coordinates_copy = list(coordinates)  # 复制剩余矩形坐标列表
    i = 1 # 用于计数
    # 循环拟合每列直线，直到所有中心坐标都被拟合完毕
    #最后得到列拟合点坐标列表 points_for_col_regression
    #得到列拟合矩形坐标列表 coordinates_for_col_regression
    while center_xy_copy:
        if i == 1:
            #print("第", i, "次拟合：")  # 第一次拟合，拟合第一列直线
            if first_point_flag:
                points_for_regression ,coordinates_for_col_regression = list(loop_col_regression(first_point, center_xy_copy, ax))  # 拟合第一列直线，返回拟合点坐标列表和拟合矩形坐标列表
            else:
                points_for_regression ,coordinates_for_col_regression = single_first_col_regression(center_xy_copy,position)  # 拟合第一列直线，返回拟合点坐标列表和拟合矩形坐标列表
            #print("第", i, "次拟合点坐标：", points_for_regression)
            #print("第", i, "次拟合矩形坐标：", coordinates_for_col_regression)
            #print("剩余待拟合点的坐标(未删除相交矩形的)：", coordinates_copy)
            #coordinates_copy_rect=remove_plots_from_points(coordinates_copy,coordinates_for_col_regression)
            #print("剩余待拟合矩形的坐标：", coordinates_copy_rect)
        else:
            #print("第", i, "次拟合：")  # 后续拟合，拟合第i列直线
            points_for_regression ,coordinates_for_col_regression= list(loop_col_regression_best(find_nth_largest_y_coordinate_vertex(center_xy_copy, 1,position), center_xy_copy, ax, position))  # 拟合第i列直线，返回拟合点坐标列表
            #print("第", i, "列拟合点坐标：", points_for_regression)
        #print("第", i, "列拟合直线的斜率:",k_list[i-1],"和截距：",b_list[i-1] )
        #判断拟合的第一列直线是否与剩余矩形框相交，如果有相交的，则将相交的矩形的中心坐标加入到本列的拟合点中
        #print("第", i, "列拟合点坐标：", points_for_regression)
        #print("第", i, "列拟合矩形坐标：", coordinates_for_col_regression)
        #print("剩余待拟合点的坐标(未删除相交矩形的)：", center_xy_copy)
        #print("剩余待拟合矩形的坐标(未删除相交矩形的)：", coordinates_copy)
        #print((points_for_regression))
        flag,center_point=check_line_intersects_rectangles(col_k_list[i-1], col_b_list[i-1],remove_plots_from_points(coordinates_copy,coordinates_for_col_regression) )
        if flag:
            #print("第", i, "列拟合的直线与剩余相交矩形的中心坐标为：",center_point)
            #print("相交矩形的矩形框为：",center_to_rectangle[center_point[0]])
            #coordinates_copy = list(remove_plots_from_points(coordinates_copy,[center_to_rectangle[center_point[0]]])) #删除相交矩形的坐标
            if position==1:
                for point in center_point:
                    if point[0]>points_for_regression[0][0]:
                        points_for_regression.extend(center_point) #将相交矩形的中心坐标加入到本列的拟合点中
                        coordinates_for_col_regression.append(center_to_rectangle[point]) #将相交矩形的中心坐标对应的矩形坐标对加入到本列的拟合矩形坐标对中
                        center_xy_copy = list(remove_plots_from_points(center_xy_copy,center_point))  
                        coordinates_copy = list(remove_plots_from_points(coordinates_copy,coordinates_for_col_regression))
            elif position==-1:
                for point in center_point:
                    if point[0]<points_for_regression[0][0]:
                        points_for_regression.extend(center_point) #将相交矩形的中心坐标加入到本列的拟合点中
                        coordinates_for_col_regression.append(center_to_rectangle[point]) #将相交矩形的中心坐标对应的矩形坐标对加入到本列的拟合矩形坐标对中 
                        center_xy_copy = list(remove_plots_from_points(center_xy_copy,center_point))  
                        coordinates_copy = list(remove_plots_from_points(coordinates_copy,coordinates_for_col_regression))
            else:
                points_for_regression.extend(center_point) #将相交矩形的中心坐标加入到本列的拟合点中
                coordinates_for_col_regression.append(center_to_rectangle[point]) #将相交矩形的中心坐标对应的矩形坐标对加入到本列的拟合矩形坐标对中 
                center_xy_copy = list(remove_plots_from_points(center_xy_copy,center_point))  
                coordinates_copy = list(remove_plots_from_points(coordinates_copy,coordinates_for_col_regression))            
        #print("第", i, "次拟合点坐标：", points_for_regression)
            #print("剩余待拟合矩形的坐标(删除相交矩形的)：", coordinates_copy)
        #else:
            #print("第", i, "列拟合的直线与剩余矩形框不相交，无遗漏矩形")
        #print("第", i, "列拟合点坐标：", points_for_regression)
        else:
            print("第", i, "列拟合的直线与剩余矩形框不相交")
        #print("第", i, "次拟合点坐标：", points_for_regression)
        final_points_for_col_regression=sort_points_by_ymax_to_ymin(points_for_regression) #对拟合的点坐标按y值从大到小排序
        #print("排序后第", i, "列拟合点坐标：", final_points_for_col_regression)
        new_points_for_col_regression=remove_duplicates(final_points_for_col_regression) #删除重复的点坐标
        #print("第", i, "次拟合点坐标：", new_points_for_col_regression)
        points_for_col_regression.append(new_points_for_col_regression)  # 将每列的拟合点坐标列表添加到总的列表中
        #print("第", i, "列拟合点坐标：", points_for_col_regression[i-1])
        center_xy_copy = list(remove_plots_from_points(center_xy_copy, points_for_regression))  # 从中心坐标列表中删除拟合过的坐标
        #print("拟合第", i, "列后的剩余点坐标：", center_xy_copy)
        #draw_line(col_k_list[i-1], col_b_list[i-1], ax, 'green',0,img_size[0])
        
        i+=1
        print('\n')
    #print("拟合完毕，得到每列的拟合点坐标列表：", points_for_col_regression)
    return points_for_col_regression


#定义对拟合的每一列进行检查的函数(主要是合并相交矩形)
#传入拟合好的每一列的拟合点坐标列表points_for_col_regression，子图ax，行数nums_of_row，摄像头位置position
#更新每一列的拟合点坐标列表points_for_col_regression
#返回更新后的center_xy
def all_col_regression_check(points_for_col_regression,ax,nums_of_row,position): 
    """
    定义对拟合的每一列进行检查的函数(主要是合并相交矩形)
    :param points_for_col_regression: 拟合好的每一列的拟合点坐标列表
    :param ax: 子图ax
    :param nums_of_row: 行数nums_of_row
    :param position: 摄像头位置
    :return: 更新后的center_xy,points_for_col_regression
    """
    #对拟合好的每一列进行检查，看是否存在同列中心坐标对应的矩形有相交的情况，若有，则将该两相交矩形的中心坐标从该列的拟合点中删除，
    #在该列拟合点坐标列表中增加由两相交矩形合并而成的矩形中心坐标，并将该矩形的中心坐标加入到该列的拟合点中，并将该矩形的坐标对加入到该列的拟合矩形坐标对中
    for i in range(len(points_for_col_regression)):
        cols_points=points_for_col_regression[i]
        col_inter_flag=False
        end_flag=False
        count=0
        while len(points_for_col_regression[i])>nums_of_row and end_flag==False:  #拟合后每列坐标点个数大于行数才进行判断是否存在相交的情况
            #print("第", i+1, "列拟合点坐标个数不等于行数，进行相交矩形检查")
            #print("第", i+1, "列拟合点坐标：", cols_points)
            if count>=20: #若相交矩形检查次数大于10次，则退出循环
                end_flag=True
                break
            count+=1
            for j in range(len(cols_points)-1): #遍历该列的拟合点坐标，判断是否存在相交的情况
                point1=cols_points[j]  #第一个点
                point2=cols_points[j+1]  #第二个点
                f,rect,intersection_area_ratio=merge_rectangles(center_to_rectangle[point1],center_to_rectangle[point2],max_intersection_area_ratio) #判断两个矩形是否有重叠部分，并计算相交面积占比
                if f :#存在相交的情况
                    col_inter_flag=True #该列存在相交的情况
                    points_for_col_regression[i]=remove_plots_from_points(points_for_col_regression[i],[point1,point2]) #将相交的两个点从该列的拟合点坐标列表中删除
                    merged_center_point=((rect[0][0]+rect[1][0])/2,(rect[0][1]+rect[1][1])/2) #计算合并后的矩形中心坐标
                    points_for_col_regression[i].append(merged_center_point) #在该列的拟合点坐标列表中增加由两相交矩形合并而成的矩形中心坐标
                    points_for_col_regression[i]=sort_points_by_ymax_to_ymin(points_for_col_regression[i]) #对该列的拟合点坐标列表按y值从大到小排序
                    center_to_rectangle[merged_center_point]=rect #将合并后的矩形的坐标对加入到中心坐标to矩形坐标对的字典中
                    rectangle_to_center[rect]=merged_center_point  #将合并后的矩形的中心坐标加入到矩形坐标对to中心坐标的字典中
                    rect1=patches.Rectangle(rect[0],rect[1][0]-rect[0][0],rect[1][1]-rect[0][1],linewidth=2,edgecolor='green',facecolor='none') #绘制合并后的矩形
                    #ax.add_patch(rect1)
                    ax.text(rect[0][0],rect[1][1],str(round(intersection_area_ratio,4)),color='red',fontsize=10) #显示相交面积占比
                    if len(points_for_col_regression[i])==nums_of_row: #若该列的拟合点坐标列表中点个数等于行数，则退出循环
                        end_flag=True
                        break
                else:
                    if j==len(cols_points)-2: #若遍历到最后一个点，则退出循环
                        #end_flag=True
                        break
                    else:
                        continue
        if col_inter_flag==False and len(points_for_col_regression[i])>nums_of_row: #若该列的拟合点坐标列表中点个数大于行数，但不存在相交的情况，则对相交面积占比大于0.005的矩形进行合并
            rect_for_col_regression=[] #用于存储该列的拟合矩形坐标对
            for point in cols_points: #将该列的拟合点坐标列表转为矩形坐标对列表
                rect_for_col_regression.append(center_to_rectangle[point])
            fff,rect1,rect2,s=find_intersect_max_area_rectangles(rect_for_col_regression) #找出两相交矩形面积最大的两个矩形
            if fff and rect1!=None and rect2!=None : #若存在两相交矩形，则进行合并
                point1=rectangle_to_center[rect1]
                point2=rectangle_to_center[rect2]
                #print(point1)
                #print(point2)
                ff,rect,intersection_area_ratio=merge_rectangles(rect1,rect2,0.005) #判断两个矩形是否有重叠部分，并计算相交面积占比
                points_for_col_regression[i]=remove_plots_from_points(points_for_col_regression[i],[point1,point2]) #将相交的两个点从该列的拟合点坐标列表中删除
                merged_center_point=((rect[0][0]+rect[1][0])/2,(rect[0][1]+rect[1][1])/2) #计算合并后的矩形中心坐标
                points_for_col_regression[i].append(merged_center_point) #在该列的拟合点坐标列表中增加由两相交矩形合并而成的矩形中心坐标
                points_for_col_regression[i]=sort_points_by_ymax_to_ymin(points_for_col_regression[i]) #对该列的拟合点坐标列表按y值从大到小排序
                center_to_rectangle[merged_center_point]=rect #将合并后的矩形的坐标对加入到中心坐标to矩形坐标对的字典中
                rectangle_to_center[rect]=merged_center_point  #将合并后的矩形的中心坐标加入到矩形坐标对to中心坐标的字典中
                rect1=patches.Rectangle(rect[0],rect[1][0]-rect[0][0],rect[1][1]-rect[0][1],linewidth=2,edgecolor='green',facecolor='none') #绘制合并后的矩形
                #ax.add_patch(rect1)
                ax.text(rect[0][0],rect[1][1],str(round(intersection_area_ratio,4)),color='red',fontsize=10) #显示相交面积占比
                if len(points_for_col_regression[i])==nums_of_row: #若该列的拟合点坐标列表中点个数等于行数，则退出循环
                    break 
    #按照每一列的第一个坐标的x值,从左侧开始对列数进行排序
    points_for_col_regression=sorted(points_for_col_regression,key=lambda x:x[0][0])
    #因为进行了合并，所以每列拟合点的数量可能大于行数，所以需要更新拟合点坐标
    #更新中心坐标列表
    center_xy=[]
    for i in range(len(points_for_col_regression)):
        for j in range(len(points_for_col_regression[i])):
            center_xy.append(points_for_col_regression[i][j])
    return center_xy,points_for_col_regression


##绘制中心点和矩形框
def draw_center_point_and_rect(center_xy,ax,img_size):
    """
    绘制中心点和矩形框
    :param center_xy: 中心点坐标列表
    :param ax: 子图ax
    :param img_size: 图像尺寸
    :return: None
    """
    for point in center_xy:
        #绘制中心点
        ax.plot(point[0], point[1], 'o', color='blue', markersize=5)
        #绘制矩形框
        rect=center_to_rectangle[point]
        x1=rect[0][0]
        y1=rect[0][1]
        width=rect[1][0]-x1
        height=rect[1][1]-y1
        rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)


#定义绘制拟合列直线的函数
def draw_col_line(col_k_list,col_b_list,points_for_col_regression,ax,img_size):
    """
    定义绘制拟合列直线的函数
    :param col_k_list: 每列的斜率列表
    :param col_b_list: 每列的截距列表
    :param points_for_col_regression: 拟合好的每一列的拟合点坐标列表
    :param ax: 子图ax
    :param img_size: 图像尺寸
    :return: None
    """
    #绘制拟合列直线
    for i in range(len(points_for_col_regression)):
        draw_line(col_k_list[i],col_b_list[i],ax,'green',0,img_size[0]) #绘制拟合的直线


# 定义绘制拟合行直线的函数
def draw_row_line(row_k_list,row_b_list,points_for_row_regression,ax,img_size):
    """
    定义绘制拟合行直线的函数
    :param row_k_list: 每行的斜率列表
    :param row_b_list: 每行的截距列表
    :param points_for_row_regression: 拟合好的每一行的拟合点坐标列表
    :param ax: 子图ax
    :param img_size: 图像尺寸
    :return: None
    """
    #绘制拟合行直线
    for i in range(len(points_for_row_regression)):
        draw_line(row_k_list[i],row_b_list[i],ax,'orange',0,img_size[0]) #绘制拟合的直线

#定义显示中心坐标的行与列的函数
def display_row_col(points_to_matrix_dict,center_xy,ax):
    """
    定义显示中心坐标的行与列的函数
    :param points_to_matrix_dict: 点坐标到矩阵坐标的字典
    :param center_xy: 中心点坐标列表
    :param ax: 子图ax
    :return: None
    """
    for point in center_xy:
        ax.text(point[0], point[1], str(points_to_matrix_dict[point]), color='black', fontsize=10)  # 绘制中心坐标对应的行列号


#定义设置坐标轴范围和保存路径等的函数
def set_axis_and_save(col_flag,ax,img_name,img_size,save_dir,save_flag): 
    """
    定义设置坐标轴范围和保存路径等的函数
    :param col_flag: 是否进行列直线拟合
    :param ax: 子图ax
    :param img_name: 图像名称    
    :param img_size: 图像尺寸
    :param save_dir: 保存路径
    :param save_flag: 是否保存图形
    :return: None
    """
    # 设置图形的标题和坐标轴标签
    ax.set_title(img_name)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # 设置坐标轴范围
    ax.set_xlim(0, img_size[0])
    ax.set_ylim(img_size[1], 0)  # 颠倒y轴坐标
    if col_flag:
        #保存图形
        save_name = img_name + ".jpg"  # 使用 img_name 作为文件名，并添加文件扩展名
        save_path = os.path.join(save_dir, save_name)
        if save_flag:
            plt.savefig(save_path)
            timer = threading.Timer(0.05, close_plot)
            timer.start()
            #进入Tkinter的主事件循环
            plt.get_current_fig_manager().window.mainloop()
            plt.close()
            plt.close('all')
        # 显示图形
    else:#列直线拟合错误，只进行列直线拟合
        #保存图形
        save_name = "行列拟合错误—"+img_name + ".jpg"  # 使用 列拟合错误—img_name 作为文件名，并添加文件扩展名
        save_path = os.path.join(save_dir, save_name)
        if save_flag:
            plt.savefig(save_path)
            timer = threading.Timer(0.05, close_plot)
            timer.start()
            #进入Tkinter的主事件循环
            plt.get_current_fig_manager().window.mainloop()
            plt.close()
            plt.close('all')
    # 显示图形,
    if not save_flag:
        plt.grid(True)
        plt.show()


#定义计算中心坐标的函数
def calculate_center_xy(coordinates):
    """
    定义计算中心坐标的函数
    :param coordinates: 矩形坐标对列表
    :return: 矩形中心坐标列表，中心坐标numpy数组，中心坐标个数
    """
    # 遍历坐标对，计算中心坐标
    for coord_pair in coordinates:
        # 提取左上角和右下角坐标
        (x1, y1), (x2, y2) = coord_pair
        # 计算矩形的中心坐标
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        center_xy.append((center_x,center_y)) # 存储中心坐标
        center_to_rectangle[(center_x,center_y)]=coord_pair # 存储中心坐标与矩形坐标对的对应关系
        rectangle_to_center[coord_pair]=(center_x,center_y) # 存储矩形坐标对与中心坐标的对应关系
    center_xy_array = np.array(center_xy)#列表转为numpy数组
    return center_xy,center_xy_array,len(center_xy) #返回中心坐标列表，中心坐标numpy数组，中心坐标个数


#当缺失第一个坐标时，对第一列进行单独拟合
def single_first_col_regression(center_xy,position):
    """
    当缺失第一个坐标时，对第一列进行单独拟合
    :param center_xy: 中心点坐标列表    
    :param position: 相机位置，-1表示摄像头在左侧，1表示摄像头在右侧
    :return: 第一列的拟合点坐标列表，第一列的拟合矩形坐标对列表
    """
    rects=[]
    first_col_points=[]
    first_col_rects=[]
    for i in range(len(center_xy)):
        rects.append(center_to_rectangle[center_xy[i]])
    if position==-1: #摄像头在左侧
        rects=sorted(rects,key=lambda x:x[0][0])#按矩形坐标对的左上角x值从小到大排序
        for i in range(len(rects)):
            if i<=4:
                first_col_points.append(rectangle_to_center[rects[i]])
        k,b=linear_regression(np.array(first_col_points))
    if position==1: #摄像头在右侧
        rects=sorted(rects,key=lambda x:x[1][0],reverse=True)#按矩形坐标对的右下角x值从大到小排序
        for i in range(len(rects)):
            if i<=4:
                first_col_points.append(rectangle_to_center[rects[i]])
        k,b=linear_regression(np.array(first_col_points))
    col_k_list.append(k)
    col_b_list.append(b)
    for i in range(len(first_col_points)):
        first_col_rects.append(center_to_rectangle[first_col_points[i]])
    return first_col_points,first_col_rects


#获取一组中心坐标对应矩形的平均宽度和高度
def get_width_and_height(points):
    """
    获取一组中心坐标对应矩形的平均宽度和高度
    :param points: 一组中心坐标
    :return: 平均宽度，平均高度
    """
    width=0
    height=0
    for point in points:
        rect=center_to_rectangle[point]
        (x1, y1), (x2, y2) = rect
        width = width + x2 - x1
        height = height + y2 - y1
    width=width/len(points)
    height=height/len(points)
    return  width,height


#绘制列直线，行直线，显示行列序号
def display_colline_rowline_and_num(col_k_list,col_b_list,row_k_list,row_b_list,points_to_matrix_dict,center_xy,ax,img_size,points_for_col_regression):
    """
    绘制列直线，行直线，显示行列序号
    :param col_k_list: 每列的斜率列表
    :param col_b_list: 每列的截距列表
    :param row_k_list: 每行的斜率列表
    :param row_b_list: 每行的截距列表
    :param points_to_matrix_dict: 点坐标到矩阵坐标的字典
    :param center_xy: 中心点坐标列表
    :param ax: 子图ax
    :param img_size: 图像尺寸
    :param points_for_col_regression: 拟合好的每一列的拟合点坐标列表
    :return: None
    """
    draw_col_line(col_k_list,col_b_list,points_for_col_regression,ax,img_size)#绘制拟合列直线
    draw_row_line(row_k_list,row_b_list,points_for_row_regression,ax,img_size)#绘制拟合行直线
    display_row_col(points_to_matrix_dict,center_xy,ax)#绘制中心坐标对应的行列号


#对center_xy对应矩形按需要进行排序（x坐标从小到大或者x坐标从大到小），返回排序后的列表
def sort_center_xy_by_rect_x_(center_xy,f):
    """
    对center_xy对应矩形按需要进行排序（x坐标从小到大或者x坐标从大到小），返回排序后的列表
    :param center_xy: 中心点坐标列表
    :param f: 1表示按x坐标从大到小排序，-1表示按x坐标从小到大排序
    :return: 排序后的中心点坐标列表
    """
    coordinates=[]
    sorted_center_xy=[]
    for i in range(len(center_xy)):
        coordinates.append(center_to_rectangle[center_xy[i]])
    if f:#按x坐标从大到小排序
        coordinates=sorted(coordinates,key=lambda x:x[1][0],reverse=True)
    else:#按x坐标从小到大排序
        coordinates=sorted(coordinates,key=lambda x:x[0][0])
    for i in range(len(coordinates)):
        sorted_center_xy.append(rectangle_to_center[coordinates[i]])
    return sorted_center_xy


#f为True时按x坐标从大到小排序，计算从大到小排序后的中心坐标，求出nums个对应的矩形坐标对的右下角横标坐标均值和方差
#f为False时按x坐标从小到大排序，计算从小到大排序后的中心坐标，求出nums个对应的矩形坐标对的左上角横标坐标均值和方差
def calculate_rectangles_x_mean_and_var(center_xy,nums):
    """
    计算从大到小排序后的中心坐标，求出nums个对应的矩形坐标对的右下角横标坐标均值和方差
    :param center_xy: 中心点坐标列表
    :param nums: 需要计算的矩形个数
    :return: 右侧均值和方差，左侧均值和方差
    """
    # 按x坐标从大到小排序,即右侧均值和方差
    rects_max_to_min=[]
    rectxs_max_to_min=[]
    sorted_center_xy_max_to_min=sort_center_xy_by_rect_x_(center_xy,True)
    for i in range(nums):
        rects_max_to_min.append(center_to_rectangle[sorted_center_xy_max_to_min[i]])    # 存储矩形坐标对
        rectxs_max_to_min.append(rects_max_to_min[i][1][0])    # 存储矩形右下角横标坐标
    x_mean_max_to_min=np.mean(rectxs_max_to_min)
    x_var_max_to_min=np.var(rectxs_max_to_min)
    # 按x坐标从小到大排序,即左侧均值和方差
    rects_min_to_max=[]
    rectxs_min_to_max=[]
    sorted_center_xy_min_to_max=sort_center_xy_by_rect_x_(center_xy,False)
    for i in range(nums):
        rects_min_to_max.append(center_to_rectangle[sorted_center_xy_min_to_max[i]])    # 存储矩形坐标对
        rectxs_min_to_max.append(rects_min_to_max[i][0][0])    # 存储矩形左上角横标坐标
    x_mean_min_to_max=np.mean(rectxs_min_to_max)
    x_var_min_to_max=np.var(rectxs_min_to_max)
    return x_mean_max_to_min,x_var_max_to_min,x_mean_min_to_max,x_var_min_to_max  #返回右侧均值和方差，左侧均值和方差


#判断摄像头位置,根据方差判断摄像头位置
#如果left_var明显小于right_var，则摄像头在左侧
# 反之right_var明显小于left_var，则摄像头在右侧
def judge_camera_position(center_xy,nums):
    """
    判断摄像头位置,根据方差判断摄像头位置
    :param center_xy: 中心点坐标列表
    :param nums: 需要计算的矩形个数
    :return: -1表示左侧摄像头，1表示右侧摄像头，0表示摄像头在中间
    """
    right_mean,right_var,left_mean,left_var=calculate_rectangles_x_mean_and_var(center_xy,nums)
    if left_var<right_var:
        return -1 #左侧摄像头
    if right_var<left_var:
        return 1 #右侧摄像头
    else:
        return 0 #摄像头在中间


#获取第一个点坐标
def get_first_point(points):
    """
    获取第一个点坐标
    :param points: 点坐标列表
    :return: 第一个点坐标 (最靠近摄像头的点)
    """
    points_copy=points.copy() #  copy points列表
    rectangles_copy=[] #  copy 矩形坐标对列表
    rectangles_right_down_copy=[] # 存储矩形右下角坐标
    for point in points_copy:
        rectangles_copy.append(center_to_rectangle[point]) # 存储矩形坐标对
        rectangles_right_down_copy.append(center_to_rectangle[point][1]) # 存储矩形右下角坐标
    # 按矩形右下角坐标按y坐标从大到小排序
    rectangles_right_down_copy=sorted(rectangles_right_down_copy,key=lambda x:x[1],reverse=True)
    # 取第一个矩形右下角坐标
    y_max_right_down_rect_point=rectangles_right_down_copy[0]
    for point in points_copy:
        if center_to_rectangle[point][1][1]==y_max_right_down_rect_point[1]:
            first_point=point
    # 转换为中心坐标
    return tuple(first_point)


#在图片中标注摄像头位置
def put_text_of_camera_position(position,center_xy,ax,numbers):
    """
    在图片中标注摄像头位置
    :param position: -1表示左侧摄像头，1表示右侧摄像头，0表示摄像头在中间
    :param center_xy: 中心点坐标列表
    :param ax: 子图ax
    :param numbers: 需要计算的矩形个数
    :return: None
    """
    right_mean,right_var,left_mean,left_var=calculate_rectangles_x_mean_and_var(center_xy,numbers)
    #在图像中标注摄像头位置
    if position==-1:
        plt.text(img_size[0]/2+100,-15,'Left Camera',fontsize=12,color='red')
    elif position==1:
        plt.text(img_size[0]/2+100,-15,'Right Camera',fontsize=12,color='red')
    else:
        plt.text(img_size[0]/2+100,-15,'Middle Camera',fontsize=12,color='red')
    #在图像空白位置显示均值和方差
    #plt.text(0,-15,'Right Mean: '+str(round(right_mean,2))+' Variance: '+str(round(right_var,2)),fontsize=12,color='red')
    #plt.text(0,-50,'Left Mean: '+str(round(left_mean,2))+' Variance: '+str(round(left_var,2)),fontsize=12,color='red')


#获取列信息
def  get_col_info(coordinates,ax,img_size):
    """
    获取列信息
    :param coordinates: 矩形坐标对列表
    :param ax: 子图ax
    :param img_size: 图像尺寸
    :return: 中心坐标列表，列数，行数,摄像头位置，每列的拟合点坐标列表
    """
#计算中心坐标
    #计算中心坐标
    center_xy,center_xy_array,nums=calculate_center_xy(coordinates)
    center_xy=list(set(center_xy)) #去除重复的中心坐标
    center_xy_array=np.array(center_xy)
    nums=len(center_xy)
    position=judge_camera_position(center_xy,5)#判断摄像头位置
    first_point=get_first_point(center_xy)
    first_point_flag=is_first_point_correct(first_point, center_xy, position)#第一个点是否正确的标志位
    points_for_col_regression=loop_all_col_regression(first_point,first_point_flag,center_xy,ax,position) #拟合所有列直线
    center_xy,points_for_col_regression=all_col_regression_check(points_for_col_regression,ax,nums_of_row,position) #检查每列直线拟合结果是否正确,合并
    numbers_of_col=len(points_for_col_regression) #列数
    number_list_of_row=[]
    for i in range(numbers_of_col):
        number_list_of_row.append(len(points_for_col_regression[i]))
    numbers_of_row=max(number_list_of_row) #行数为最大的列数
    return center_xy , numbers_of_col, numbers_of_row,position,points_for_col_regression#返回中心坐标，列数，行数,摄像头位置


#给点若干点points的坐标,找出线性拟合的离群点
def find_outlier_points(points):
    """
    给点若干点points的坐标,找出线性拟合的离群点
    :param points: 点坐标列表
    :return: 离群点列表
    """
    k,b=linear_regression(np.array(points)) #对points进行直线拟合
    #draw_line(k,b,ax,'red',img_size[0],0) #绘制拟合直线
    outlier_points=[] #定义一个空列表,用于存储离群点
    for point in points:
        if abs(point[1]-k*point[0]-b)>0.5*math.sqrt(sum((point[1]-k*point[0]-b)**2 for point in points)): #如果点与拟合直线距离大于3倍标准差
            outlier_points.append(point) #将离群点添加到列表中
            #print("离群点坐标为：",point)
    return outlier_points #返回离群点列表


#判断某点point是否在列表points中,如果在,返回True,否则返回False
def is_point_in_list(point,points):
    """
    判断某点point是否在列表points中,如果在,返回True,否则返回False
    :param point: 点坐标
    :param points: 点坐标列表
    :return: True or False
    """
    flag=False
    for p in points:
        if p[0]==point[0] and p[1]==point[1]:
            flag=True
            break
    return flag


#判断某一行(num+1)有无离群点，如果有,则将离群点从该列的点列表中移除,并找出在哪一列的哪个位置，并且确定存在离群点的该行是否有缺失的点
def  judge_row_outline_points(points_for_col_regression,num):
    """
    判断某一行(num+1)有无离群点，如果有,则将离群点从该列的点列表中移除,并找出在哪一列的哪个位置，并且确定存在离群点的该行是否有缺失的点
    :param points_for_col_regression: 每列的拟合点坐标列表
    :param num: 行号
    :return: 该行的点坐标列表和该行是否有缺失的点的标志位 和 离群点所在的列号和序号
    """
    points_for_col_regression_copy=points_for_col_regression.copy() #  copy points_for_col_regression列表
    points_per_row=[] #定义一个列表,用于存储每行的点坐标
    flag_per_row=False #该行是否有缺失的点的标志位,默认没有缺失的点
    for i in range(nums_of_col):
        points_per_row.append(points_for_col_regression_copy[i][num])
    outline_points=find_outlier_points(points_per_row) #找出离群点
    if len(outline_points)>0: #如果有离群点
        #print("第",num+1,"行存在离群点:",outline_points)
        col_num=[] #定义一个列表,用于存储离群点所在的列号
        point_num=[] #定义一个列表,用于存储离群点在该列的序号
        for j in range(nums_of_col) : #遍历每列的点,如果该列的点数小于行数,判断离群点是不是在该列当中
            if len(points_for_col_regression_copy[j])<nums_of_row:    
                for point in outline_points: #遍历离群点列表
                    if is_point_in_list(point,points_for_col_regression_copy[j]): #如果离群点在该列中,则得到该列的列号和该点在该列的序号
                        #print("第",num+1,"行的离群点",point,"在第",j+1,"列中") 
                        col_num.append(j)   #将该列的列号添加到列表中
                        point_num.append(points_for_col_regression_copy[j].index(point))  #将该点在该列的序号添加到列表中
        #print("第",num+1,"行的离群点所在的列号为:",col_num)
        #print("第",num+1,"行的离群点在该列的序号为:",point_num)
        flag=False #定义一个标志位,用于判断该列是否有缺失的点
        for k in range(len(col_num)): 
            if len(points_for_col_regression_copy[col_num[k]])<nums_of_row: #如果该列的点数小于行数,则表明该列有缺失的点
                flag=True #该列有缺失的点的标志位为True
                break
    if (len(points_per_row)-len(outline_points))<nums_of_col and  flag==True: #如果该行的点数小于列数,则表明该行有缺失的点
        #print("第",num+1,"行有缺失的点,在第","(",col_num,"+1)列中")
        for point in outline_points: #遍历离群点列表
            points_per_row.remove(point) #将离群点从该行拟合列表中移除
        flag_per_row=True #该行有缺失的点的标志位为True

    return points_per_row,flag_per_row,col_num,point_num #返回该行的点坐标列表和该行是否有缺失的点的标志位 和 离群点所在的列号和序号


#把point插入到列表points中的指定位置num,并保持列表的有序性
def insert_point_to_list(point,points,num):
    """
    把point插入到列表points中的指定位置num,并保持列表的有序性
    :param point: 点坐标
    :param points: 点坐标列表
    :param num: 指定位置
    :return: 插入后的点坐标列表
    """
    points_copy=points.copy() #  copy points列表
    points_list=[] #定义一个空列表,用于存储插入后的点坐标
    for i in range(len(points_copy)):
        if i==num:
            points_list.append(point)
        points_list.append(points_copy[i])
    return points_list #返回插入后的点坐标列表


#获得用以补充的矩形的周围的样本点坐标列表
def get_fill_rectangle_sample_points(points_for_col_regression,col_num,row_num):
    """
    获得用以补充的矩形的周围的样本点坐标列表
    :param points_for_col_regression: 每列的拟合点坐标列表
    :param col_num: 补充矩形所在的列号
    :param row_num: 补充矩形所在的行号
    :return: 补充矩形的样本点坐标列表
    """
    sample_points=[] #定义一个空列表,用于存储补充矩形的样本点坐标
    #获得col_num列的点列表中,row_num行的点周围三个点(位于中间时)或两个点(位于四个顶点)的坐标
    if row_num>0 and row_num<nums_of_row-1: #如果不是第一行或最后一行
        sample_points.append(points_for_col_regression[col_num][row_num-1]) #将该点的上一行的点添加到样本点列表中
        sample_points.append(points_for_col_regression[col_num][row_num+1]) #将该点的下一行的点添加到样本点列表中
        if col_num==0: #如果是第一列
            sample_points.append(points_for_col_regression[col_num+1][row_num]) #将该点的右一列的点添加到样本点列表中
        if col_num==nums_of_col-1: #如果是最后一列
            sample_points.append(points_for_col_regression[col_num-1][row_num]) #将该点的左一列的点添加到样本点列表中
        else: #既不是第一列也不是最后一列
            sample_points.append(points_for_col_regression[col_num-1][row_num]) #将该点的左一列的点添加到样本点列表中
            sample_points.append(points_for_col_regression[col_num+1][row_num]) #将该点的右一列的点添加到样本点列表中
    elif row_num==0: #如果是第一行
        sample_points.append(points_for_col_regression[col_num][row_num+1]) #将该点的下一行的点添加到样本点列表中
        if col_num==0: #如果是第一列
            sample_points.append(points_for_col_regression[col_num+1][row_num]) #将该点的右一列的点添加到样本点列表中
        if col_num==nums_of_col-1: #如果是最后一列
            sample_points.append(points_for_col_regression[col_num-1][row_num]) #将该点的左一列的点添加到样本点列表中
        else: #既不是第一列也不是最后一列
            sample_points.append(points_for_col_regression[col_num-1][row_num]) #将该点的左一列的点添加到样本点列表中
            sample_points.append(points_for_col_regression[col_num+1][row_num]) #将该点的右一列的点添加到样本点列表中
    elif row_num==nums_of_row-1 : #如果是最后一行
        sample_points.append(points_for_col_regression[col_num][row_num-1]) #将该点的上一行的点添加到样本点列表中
        if col_num==0: #如果是第一列
            sample_points.append(points_for_col_regression[col_num+1][row_num]) #将该点的右一列的点添加到样本点列表中
        if col_num==nums_of_col-1: #如果是最后一列
            sample_points.append(points_for_col_regression[col_num-1][row_num]) #将该点的左一列的点添加到样本点列表中
        else: #既不是第一列也不是最后一列
            sample_points.append(points_for_col_regression[col_num-1][row_num]) #将该点的左一列的点添加到样本点列表中
            sample_points.append(points_for_col_regression[col_num+1][row_num]) #将该点的右一列的点添加到样本点列表中
    return sample_points #返回补充矩形的样本点坐标列表


#获得补充矩形的坐标对数据
def get_fill_rectangle_data(center_xy,sample_points):
    """
    获得补充矩形的坐标对数据
    :param center_xy: 中心点坐标列表
    :param sample_points: 补充矩形的样本点坐标列表
    :return: 补充矩形的坐标对数据
    """
    width,height=get_width_and_height(sample_points) #计算补充矩形的宽度和高度
    x1,y1=center_xy[0]-width/2,center_xy[1]-height/2 #计算矩形左上角坐标
    x2,y2=center_xy[0]+width/2,center_xy[1]+height/2 #计算矩形右下角坐标
    coordinate=((x1,y1),(x2,y2)) #定义矩形坐标对数据
    return coordinate


#依据position摄像头位置,将列拟合点坐标按照x坐标排序（从小到大）
def sort_col_regression_points_by_x(points_for_col_regression,position):
    """
    依据position摄像头位置,将列拟合点坐标按照x坐标排序（从小到大）
    :param points_for_col_regression: 每列的拟合点坐标列表
    :param position: 摄像头位置
    :return: 排序后的点坐标列表
    """
    if position==1: #如果摄像头在右侧
        points_for_col_regression_sorted=[] #定义一个空列表,用于存储排序后的点坐标
        for i in range(len(points_for_col_regression)):
            points_for_col_regression_sorted.append(points_for_col_regression[numbers_of_col-1-i])
    else: #如果摄像头在左侧或中央
        points_for_col_regression_sorted=points_for_col_regression.copy() #  copy points_for_col_regression列表
    return points_for_col_regression_sorted #返回排序后的点坐标列表


#依据position摄像头位置,将列拟合的直线参数k,b从左到右排序
def sort_col_regression_line_params(col_k_list,col_b_list,position):
    """
    依据position摄像头位置,将列拟合的直线参数k,b从左到右排序
    :param col_k_list: 每列的拟合直线的k值列表
    :param col_b_list: 每列的拟合直线的b值列表
    :param position: 摄像头位置
    :return: 排序后的k值和b值列表
    """
    if position==1: #如果摄像头在右侧
        col_k_list_sorted=[] #定义一个空列表,用于存储排序后的k值
        col_b_list_sorted=[] #定义一个空列表,用于存储排序后的b值
        for i in range(len(col_k_list)):
            col_k_list_sorted.append(col_k_list[numbers_of_col-1-i])
            col_b_list_sorted.append(col_b_list[numbers_of_col-1-i])
    else: #如果摄像头在左侧或中央
        col_k_list_sorted=col_k_list.copy() #  copy col_k_list列表
        col_b_list_sorted=col_b_list.copy() #  copy col_b_list列表
    return col_k_list_sorted,col_b_list_sorted #返回排序后的k值和b值列表


#通过不断尝试拟合行直线，判断列有无缺失的点，
def check_col_regression_by_row_fitting(center_xy,points_for_col_regression,ax,position):
    """
    通过不断尝试拟合行直线，判断列有无缺失的点，
    :param center_xy: 中心点坐标列表
    :param points_for_col_regression: 每列的拟合点坐标列表
    :param ax: 绘图对象
    :param position: 摄像头位置
    :return: 所有行列拟合结果是否正确的标志位
    """
    points_for_row_regression=[] #定义一个列表,用于存储每行的拟合点坐标
    coordinates=[] #定义一个列表,用于存储拟合矩形的坐标对
    center_to_rectangle={} #定义一个字典,用于存储中心点坐标和对应的拟合矩形坐标对
    center_xy_copy=center_xy.copy() #  copy center_xy列表
    points_for_col_regression_copy=sort_col_regression_points_by_x(points_for_col_regression,position) #排序列拟合点坐标列表
    col_k_list_sorted,col_b_list_sorted=sort_col_regression_line_params(col_k_list,col_b_list,position) #对列直线拟合结果进行排序
    point_rows=[] #定义一个列表,用于存储每行的点坐标
    for i in range(nums_of_row): #遍历每行的点
        points_per_row,flag_lack_per_row,col_per_num,point_per_num=judge_row_outline_points(points_for_col_regression_copy,i) #判断第i行有无离群点
        if flag_lack_per_row: #如果该行有缺失的点
            points_per_col_list=[] #定义一个空列表
            per_row_k,per_row_b=linear_regression(np.array(points_per_row)) #对该行的点进行直线拟合,得到该行的k和b
            for j in range(len(col_per_num)): #遍历离群点所在的列号
                fill_points=[] #定义一个空列表,用于存储补充矩形的样本点坐标
                col_k,col_b=col_k_list_sorted[col_per_num[j]],col_b_list_sorted[col_per_num[j]] #得到该列的k和b
                #计算列直线与行直线的交点，得到缺失点的坐标
                x=abs(int( (-col_b+per_row_b)/(per_row_k-col_k)))
                y=abs(int( per_row_k*x+per_row_b))
                lack_point=tuple([x,y])
                #print("第",i+1,"行的缺失点在第",col_per_num[j]+1,"列,坐标为：",lack_point)
                points_per_col_list=insert_point_to_list(lack_point,points_for_col_regression_copy[col_per_num[j]],point_per_num[j]) #把缺失点添加到该列的点列表中的第point_per_num[j]个位置
                points_per_row_list=insert_point_to_list(lack_point,points_per_row,col_per_num[j]) #把缺失点添加到该行的点列表中的第col_per_num[j]个位置
                center_xy.append(lack_point) #将缺失点添加到中心坐标列表中
                points_for_col_regression_copy[col_per_num[j]]=points_per_col_list #更新该列的点列表
                sample_points=get_fill_rectangle_sample_points(points_for_col_regression_copy,col_per_num[j],point_per_num[j]) #获得补充矩形的样本点坐标列表
                #print("插值样本点：",sample_points)
                coordinate=get_fill_rectangle_data(lack_point,sample_points) #计算补充矩形的坐标对数据
                #print( "补充矩形的坐标对为：",coordinate)
                center_to_rectangle[lack_point]=coordinate #将缺失点的坐标和对应的矩形坐标对添加到字典中
                rectangle_to_center[coordinate]=lack_point #将矩形坐标对和对应的缺失点的坐标添加到字典中
                coordinates.append(coordinate) #将补充矩形的坐标对添加到拟合矩形列表中
                #print("补充缺失点后第",col_per_num[j]+1,"列的拟合点坐标有",len(points_per_col_list),"个点,为：",points_per_col_list)
                plt.plot(lack_point[0],lack_point[1],'ro',markersize=10) #绘制缺失点
                #draw_line(per_row_k,per_row_b,ax,'orange',img_size[0],0) #绘制该行的直线
                #print("补充缺失点后第",i+1,"行的拟合点坐标有",len(points_per_row_list),"个点,为：",points_per_row_list)
                points_for_row_regression.append(points_per_row_list) #将该行的点坐标添加到列表中
        else: #如果该行没有缺失的点
            per_row_k,per_row_b=linear_regression(np.array(points_per_row)) #对该行的点进行直线拟合,得到该行的k和b
            #draw_line(per_row_k,per_row_b,ax,'orange',img_size[0],0) #绘制该行的直线
            #print("第",i+1,"行的拟合点坐标有",len(points_per_row),"个点,为：",points_per_row)
            points_for_row_regression.append(points_per_row) #将该行的点坐标添加到列表中
    points_for_col_regression=points_for_col_regression_copy.copy() #更新列拟合点坐标列表
    #再次检查行列拟合结果是否正确
    all_right_flag=True #定义一个标志变量,用于判断所有行列拟合结果是否正确,默认正确
    for i in range(nums_of_row):
        if len(points_for_row_regression[i])!=nums_of_col: #如果该行的点数不等于列数
            all_right_flag=False #设置标志变量为错误
            #print("第",i+1,"行的点数不等于列数")
    for i in range(nums_of_col):
        if len(points_for_col_regression[i])!=nums_of_row: #如果该列的点数不等于行数
            all_right_flag=False #设置标志变量为错误
            #print("第",i+1,"列的点数不等于行数")
    if all_right_flag: #如果所有行列拟合结果正确
        print("所有行列拟合结果正确")
    return all_right_flag


#如果所有行列拟合结果正确，则进行正式的列直线与行直线的拟合，保存拟合直线参数，计算行列矩阵
def get_row_and_col_matrix(points_for_col_regression,points_for_row_regression,position):
    """
    如果所有行列拟合结果正确，则进行正式的列直线与行直线的拟合，保存拟合直线参数，计算行列矩阵
    :param points_for_col_regression: 每列的拟合点坐标列表
    :param points_for_row_regression: 每行的拟合点坐标列表
    :param position: 摄像头位置
    :return: 列直线参数,行直线参数,行列矩阵,点坐标到矩阵索引的字典
    """
    points_for_col_regression=list(sort_col_regression_points_by_x(points_for_col_regression,position)) #对列拟合点坐标列表进行排序
    col_kk_list=[] #定义一个空列表,用于存储列直线的k值
    col_bb_list=[] #定义一个空列表,用于存储列直线的b值
    row_kk_list=[] #定义一个空列表,用于存储行直线的k值
    row_bb_list=[] #定义一个空列表,用于存储行直线的b值
    for i in range(nums_of_col): #遍历每列的点
        col_points=points_for_col_regression[i] #得到第i列的点坐标列表
        col_k,col_b=linear_regression(np.array(col_points)) #对第i列的点进行直线拟合,得到该列的k和b
        col_kk_list.append(col_k) #将该列的k值添加到列表中
        col_bb_list.append(col_b) #将该列的b值添加到列表中
    print(nums_of_row)
    for i in range(nums_of_row): #遍历每行的点
        row_points=points_for_row_regression[i] #得到第i行的点坐标列表
        row_k,row_b=linear_regression(np.array(row_points)) #对第i行的点进行直线拟合,得到该行的k和b
        row_kk_list.append(row_k) #将该行的k值添加到列表中
        row_bb_list.append(row_b) #将该行的b值添加到列表中
    col_k_list_sorted,col_b_list_sorted=sort_col_regression_line_params(col_kk_list,col_bb_list,position) #对列直线拟合结果进行排序
    row_k_list_sorted,row_b_list_sorted=sort_col_regression_line_params(row_kk_list,row_bb_list,position) #对行直线拟合结果进行排序
    col_line_params=(col_k_list_sorted,col_b_list_sorted) #列直线参数
    row_line_params=(row_k_list_sorted,row_b_list_sorted) #行直线参数
    Matrix= construct_coordinate_matrix(points_for_row_regression,points_for_col_regression) #计算行列矩阵,Matrix是一个numpy数组,M[i][j]表示第i行+第j+1列的点的坐标
    points_to_matrix_dict=create_coordinate_dictionary(Matrix) #创建坐标到矩阵索引的字典,points_to_matrix_dict[point]=matrix_index,matrix_index=i*nums_of_col+j,i表示行号,j表示列号,point表示点的坐标,matrix_index表示矩阵的索引,
    return col_line_params,row_line_params,Matrix,points_to_matrix_dict #返回列直线参数,行直线参数,行列矩阵,点坐标到矩阵索引的字典


#单独拟合最靠近摄像头一列的直线，用以检查第一个点是否缺失
def fit_first_col_line_only(center_xy,ax,position):
    """
    单独拟合最靠近摄像头一列的直线，用以检查第一个点是否缺失
    :param center_xy: 中心点坐标列表
    :param ax: 绘图对象
    :param position: 摄像头位置
    :return: 第一列的点数
    """
    center_xy_cp=center_xy.copy() #  copy center_xy列表
    first_col_line_points_for_regression=[] #定义一个空列表,用于存储最靠近摄像头的列的三个点坐标
    first_col_num=0 #定义一个变量,用于存储第一列的点数
    if position==1: #如果摄像头在右侧
        center_xy_cp=sorted(center_xy_cp,key=lambda x:x[0],reverse=True) #对中心点坐标按照x坐标进行排序,从右到左
        #取前三个点出来
        for i in range(3):
            first_col_line_points_for_regression.append(center_xy_cp[i])
    else: #如果摄像头在左侧或中央
        center_xy_cp=sorted(center_xy_cp,key=lambda x:x[0]) #对中心点坐标按照x坐标进行排序,从左到右
        #取前三个点出来
        for i in range(3):
            first_col_line_points_for_regression.append(center_xy_cp[i])
    #对前三个点进行直线拟合
    k,b=linear_regression(np.array(first_col_line_points_for_regression))
    #绘制拟合直线
    #draw_line(k,b,ax,'red',img_size[0],0)
    #判断拟合的直线穿过了多少个矩形，进而判断第一列有多少个点
    for point in center_xy_cp:
        rectangle=center_to_rectangle[point] #得到该点对应的矩形坐标对
        if is_line_intersect_rectangle(k,b,rectangle): #如果拟合的直线穿过了矩形
            first_col_num+=1 #则该列的点数加1
    #print("最靠近摄像头的列的点数为：",first_col_num)
    #plt.text(first_col_line_points_for_regression[0][0]+30,first_col_line_points_for_regression[0][1]+30,str(first_col_num),fontsize=12,color='green') #绘制最靠近摄像头的列的点数
    return first_col_num #返回第一列的点数


#错误列处理
def error_col_process():
    """
    错误列处理
    :return:
    """
    center_xy,nums_of_col,nums_of_row,position,points_for_col_regression=get_col_info(coordinates,ax,img_size) #调用函数
    #print("摄像头位置为3：",position) #打印输出摄像头位置
    #print("列数为：",nums_of_col)#打印输出列数
    #print("行数为：",nums_of_row)#打印输出行数
    if len(center_xy)<30-10: #如果中心点坐标数小于30-10,则认为摄像头位置不正确,退出程序
        #print("中心点个数为：",len(center_xy)) #打印输出中心点坐标个数
        draw_center_point_and_rect(center_xy,ax,img_size) #绘制中心点和矩形框
        set_axis_and_save(0,ax,img_name,img_size,save_dir,save_flag)#设置坐标轴范围和保存路径等
        put_text_of_camera_position(position,center_xy,ax,5) #绘制摄像头位置
        exit()
        #print("检测到的中心点数过少,疑似检测程序出错,请检查")
        return
    #检查所有行列拟合结果是否正确
    #for i in range(len(points_for_col_regression)):
        #print("第",i+1,"列的拟合点有：",len(points_for_col_regression[i]),"个坐标为：",points_for_col_regression[i])
    all_right_flag =check_col_regression_by_row_fitting(center_xy,points_for_col_regression,ax,position)
    if all_right_flag: #如果所有行列拟合结果正确
        col_line_params,row_line_params,Matrix,points_to_matrix_dict=get_row_and_col_matrix(points_for_col_regression,points_for_row_regression,position) #计算行列矩阵
        display_colline_rowline_and_num( col_line_params[0],col_line_params[1],row_line_params[0],row_line_params[1],points_to_matrix_dict,center_xy,ax,img_size,points_for_col_regression) #绘制拟合直线和数字
        put_text_of_camera_position(position,center_xy,ax,5) #绘制摄像头位置
        draw_center_point_and_rect(center_xy,ax,img_size) #绘制中心点和矩形框
        set_axis_and_save(1,ax,img_name,img_size,save_dir,save_flag)#设置坐标轴范围和保存路径等
        exit()
        return
    else: #如果所有行列拟合结果不正确,则退出程序
    #print("所有行列拟合结果不正确,请检查")
        draw_col_line(col_k_list,col_b_list,points_for_col_regression,ax,img_size)
        draw_center_point_and_rect(center_xy,ax,img_size) #绘制中心点和矩形框
        put_text_of_camera_position(position,center_xy,ax,5) #绘制摄像头位置
        set_axis_and_save(0,ax,img_name,img_size,save_dir,save_flag)#设置坐标轴范围和保存路径等
        exit()
        return



#主程序
if __name__ == '__main__':
    file_path__="../col_and_row_regression/test-data/labels/vacant2_1413.txt"
    save_path__='../col_and_row_regression/test_result'
    all_img_flag__=False
    init_global_var(all_img_flag_in=all_img_flag__,file_path_in=file_path__,save_path_in=save_path__,count=1)
    center_xy,center_xy_array,center_xy_nums=calculate_center_xy(coordinates)  #调用函数,计算中心点坐标,中心点坐标数组,中心点坐标个数
    center_xy_cp=center_xy.copy() #  copy center_xy列表
    position=judge_camera_position(center_xy,5) #判断摄像头位置
    first_point=get_first_point(center_xy) #获取第一个点(最靠近摄像头的点)
    #plt.plot(first_point[0],first_point[1],marker='o',markersize=12,color='green') #绘制第一个点
    is_first_point_correct_flag=is_first_point_correct(first_point,center_xy,position) #检查第一个点是否正确
    #plt.text(first_point[0]+30,first_point[1]+30,is_first_point_correct_flag,fontsize=12,color='green') #绘制第一个点的序号
    #print("摄像头位置为：",position) #打印输出摄像头位置
    #print("中心点坐标个数为：",center_xy_nums) #打印输出中心点坐标个数
    #print("第一个点为：" ,first_point) #打印输出第一个点
    #print("第一个点是否正确：" ,is_first_point_correct_flag)
    #1.如果中心点坐标数小于30-10,则认为摄像头位置不正确,退出程序
    if len(center_xy)<30-10: 
        #print("中心点个数为：",len(center_xy)) #打印输出中心点坐标个数
        draw_center_point_and_rect(center_xy,ax,img_size) #绘制中心点和矩形框
        set_axis_and_save(0,ax,img_name,img_size,save_dir,save_flag)#设置坐标轴范围和保存路径等
        put_text_of_camera_position(position,center_xy,ax,5) #绘制摄像头位置
        exit()
    #print("1.0 中心点坐标数小于30-10,则认为摄像头位置不正确,退出程序")
    #print("检测到的中心点数过少,疑似检测程序出错,请检查")
    first_col_num=fit_first_col_line_only(center_xy_cp,ax,position) #只拟合最靠近摄像头一列的直线,获得第一列的点数
    #print("第一列点数为：",first_col_num) #打印输出第一列点数
    center_xy_cp1 , numbers_of_col, numbers_of_row,position,points_for_col_regression_cp1=get_col_info(coordinates,ax,img_size) #调用函数,获得列数,行数,列坐标列表,行坐标列表
    col_flag_scores=[] #存储每列的拟合结果的分数
    for i in range(len(points_for_col_regression_cp1)):
        #print("第",i+1,"列的拟合点有：",len(points_for_col_regression_cp1[i]),"个坐标为：",points_for_col_regression_cp1[i])
        if len(points_for_col_regression_cp1[i])==6: #如果该列的点数等于6,说明该列的拟合结果可能正确
            col_flag_scores.append(1) #标记该列的拟合结果正确
        else:
            col_flag_scores.append(0) #标记该列的拟合结果不正确
    init_global_var(all_img_flag_in=all_img_flag__,file_path_in=file_path__,save_path_in=save_path__,count=2)
    #2.如果第一个点正确,且第一列点数大于等于行数6,且中心点坐标个数大于等于30,拟合直线列数为5,且所有列的拟合结果正确,则认为所有行列拟合结果可以正确
    if is_first_point_correct_flag==1 and first_col_num>=6 and len(center_xy_cp1)>=30 and len(points_for_col_regression_cp1)==5 and sum(col_flag_scores)==5: 
        #print("2.0 如果第一个点正确,且第一列点数大于等于行数6,且中心点坐标个数小于30-10,拟合直线列数为5,且所有列的拟合结果正确,则认为所有行列拟合结果可能正确")
        center_xy,nums_of_col,nums_of_row,position,points_for_col_regression=get_col_info(coordinates,ax,img_size) #调用函数
        #print("列数为：",nums_of_col)#打印输出列数
        #print("行数为：",nums_of_row)#打印输出行数
        #检查所有行列拟合结果是否正确
        all_col_flags=[] #存储所有列的拟合结果是否正确
        for i in range(len(points_for_col_regression)):
            #print("第",i+1,"列的拟合点有：",len(points_for_col_regression[i]),"个坐标为：",points_for_col_regression[i])
            if len(points_for_col_regression[i])==nums_of_row: #如果拟合点数等于行数,说明该列的拟合结果正确
                all_col_flags.append(1) #标记该列的拟合结果正确
            else:
                all_col_flags.append(0) #标记该列的拟合结果不正确
        #2.1如果所有行列拟合结果正确
        if sum(all_col_flags)==nums_of_col:
            #print("2.1所有行列拟合结果正确")
            #获取行拟合点的坐标
            points_for_row_regression=[]
            for i in range(nums_of_row):
                row_points=[]
                for j in range(nums_of_col):
                    row_points.append(points_for_col_regression[j][i])
                points_for_row_regression.append(row_points)
            #print("points_for_row_regression:",points_for_row_regression)
            col_line_params,row_line_params,Matrix,points_to_matrix_dict=get_row_and_col_matrix(points_for_col_regression,points_for_row_regression,position) #计算行列矩阵
            display_colline_rowline_and_num( col_line_params[0],col_line_params[1],row_line_params[0],row_line_params[1],points_to_matrix_dict,center_xy,ax,img_size,points_for_col_regression) #绘制拟合直线和数字
            put_text_of_camera_position(position,center_xy,ax,5) #绘制摄像头位置
            draw_center_point_and_rect(center_xy,ax,img_size) #绘制中心点和矩形框
            set_axis_and_save(1,ax,img_name,img_size,save_dir,save_flag)#设置坐标轴范围和保存路径等
            exit()
        #2.1如果所有行列拟合结果不正确,则退出程序
        else: 
            #print("2.2 所有列拟合结果不正确,请检查")
            for j in range(len(col_k_list)):
                draw_line(col_k_list[j],col_b_list[j],ax,'green',0,img_size[0]) #绘制拟合直线
            draw_center_point_and_rect(center_xy,ax,img_size) #绘制中心点和矩形框
            put_text_of_camera_position(position,center_xy,ax,5) #绘制摄像头位置
            set_axis_and_save(0,ax,img_name,img_size,save_dir,save_flag)#设置坐标轴范围和保存路径等
            exit()
        
    #3.如果第一个点不正确,或第一列点数小于6,或中心点坐标个数大于等于30-10,或所有列的拟合结果不正确
    else:
        center_xy,nums_of_col,nums_of_row,position,points_for_col_regression=get_col_info(coordinates,ax,img_size) #调用函数
        #for i in range(nums_of_col):  # 遍历每一列
        #    print("第",i+1,"列的拟合点有：",len(points_for_col_regression[i]),"个坐标为：",points_for_col_regression[i])
        for i in range(nums_of_row):  # 遍历每一行
            maybe_row_line_points=[]  # 存储可能的行直线的点坐标列表
            true_row_line_points=[]  # 存储正确的行直线的点坐标列表
            for j in range(nums_of_col):
                if len(points_for_col_regression[j])==nums_of_row:  # 如果该列的点数等于行数,说明该列的拟合结果正确
                    true_row_line_points.append(points_for_col_regression[j][i])    # 存储正确的点坐标
                else:
                    if len(points_for_col_regression[j])>=i+1:  # 如果该列的点数大于等于该行的序号,说明该列的拟合结果可能正确
                        maybe_row_line_points.append(points_for_col_regression[j][i])    # 存储可能的点坐标
                    else:  # 如果该列的点数小于该行的序号,说明该列的拟合结果不正确,需要补充缺失的点
                        maybe_row_line_points.append(points_for_col_regression[j][0])    # 取该列第一个点作为补充点
            row_k,row_b=linear_regression(np.array(true_row_line_points))  # 拟合行直线的斜率和截距
            for point in maybe_row_line_points:  # 遍历可能的点坐标
                if is_line_intersect_rectangle(row_k,row_b,center_to_rectangle[point]):  # 判断该行直线是否穿过可能的点的矩形框
                    #print("可能的点坐标：",point)
                    continue
                else:  # 如果该行直线不穿过该点的矩形框,则认为该点是错误的
                    #print("错误的点坐标：",point)
                    for k in range(len(points_for_col_regression)) :  # 遍历每列，找到错误的点属于哪一列
                        if point in points_for_col_regression[k]:  # 如果该点属于该列
                            #print("该点属于第",k+1,"列")
                            col_k,col_b=linear_regression(np.array(points_for_col_regression[k]))  # 计算该列直线,计算两条直线的交点
                            missing_point_x=int(abs((-col_b+row_b)/(col_k-row_k)))  # 计算缺失的点的横坐标
                            missing_point_y=int(abs(col_k*missing_point_x+col_b))  # 计算缺失的点的纵坐标
                            missing_point=(missing_point_x,missing_point_y)  # 缺失的点的坐标
                            #print("缺失的点坐标：",missing_point,"属于第",k+1,"列 第",i+1,"行")
                            if i>=len(points_for_col_regression[k]):  # 如果该行的序号大于等于该列的点数,说明缺失的点在最后一个
                                points_for_col_regression[k].append(missing_point)
                            else:
                                points_for_col_regression[k]=insert_point_to_list(missing_point,points_for_col_regression[k],i)  # 插入缺失的点到该列的点列表中
                            #print("该列的点坐标：",points_for_col_regression[k])
                            center_xy.append(missing_point) #将缺失点加入中心点坐标列表中
                            sample_points=true_row_line_points  # 取该行的点坐标作为参考点
                            coordinate=get_fill_rectangle_data(missing_point,sample_points) #计算补充矩形的坐标对数据
                            center_to_rectangle[missing_point]=coordinate #将缺失点的坐标对数据加入中心点坐标对字典中
                            rectangle_to_center[coordinate]=missing_point #将缺失点的坐标对数据加入矩形坐标对字典中
                            coordinates.append(coordinate) #将缺失点的坐标对数据加入坐标对列表中
        center_xy,points_for_col_regression=all_col_regression_check(points_for_col_regression,ax,nums_of_row,position) #检查每列直线拟合结果是否正确,合并
        for jjj in range(nums_of_col):
            points_for_col_regression[jjj]=sorted(points_for_col_regression[jjj],key=lambda x:x[1],reverse=True)  # 按横坐标排序每列的点y坐标,这里应该是按纵坐标排序,从大到小
        all_col_line_flags=[]  # 用于存储每一列的线性回归是否成功的标志列表
        for iii in range(nums_of_col):
            if len(points_for_col_regression[iii])==nums_of_row:
                all_col_line_flags.append(1)
        if sum(all_col_line_flags)==nums_of_col:  # 如果所有列的拟合都成功
            print("所有列拟合都成功")
            points_for_row_regression=[]
            for i in range(nums_of_row):
                row_points=[]
                for j in range(nums_of_col):
                    row_points.append(points_for_col_regression[j][i])
                points_for_row_regression.append(row_points)
            col_line_params,row_line_params,Matrix,points_to_matrix_dict=get_row_and_col_matrix(points_for_col_regression,points_for_row_regression,position) #计算行列矩阵
            display_colline_rowline_and_num( col_line_params[0],col_line_params[1],row_line_params[0],row_line_params[1],points_to_matrix_dict,center_xy,ax,img_size,points_for_col_regression) #绘制拟合直线和数字
            put_text_of_camera_position(position,center_xy,ax,5) #绘制摄像头位置
            draw_center_point_and_rect(center_xy,ax,img_size) #绘制中心点和矩形框
            set_axis_and_save(1,ax,img_name,img_size,save_dir,save_flag)#设置坐标轴范围和保存路径等
            exit()
        else:#存在有列拟合错误的情况
            print("有列拟合失败")
            coordinates,img_size=read_rectangles_from_txt(img_path) # 存储矩形框坐标对列表和图像大小
            init_global_var(all_img_flag_in=all_img_flag__,file_path_in=file_path__,save_path_in=save_path__,count=2) #初始化全局变量
            error_col_process() #调用错误列处理函数
            exit()