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
    nums_of_row=8
    #设定列数(暂时只针对6*5的行列进行拟合)
    nums_of_col=4
    numbers_of_col=4#列数
    numbers_of_row=8#行数
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
        print("1斜率为0，为列直线")
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






#定义合并一条列直线中相交的矩形的函数
def merge_col_regression_intersect_rectangles(center_xy,points_for_per_col_regression,ax,position):
    """
    定义合并一条列直线中相交的矩形的函数
    :param points_for_per_col_regression: 每列的拟合点坐标列表
    :param ax: 子图ax
    :param position: 摄像头位置
    :return: 得到每列的拟合点坐标列表 points_for_col_regression
    """
    col_points=points_for_per_col_regression 
    col_inter_flag=False
    end_flag=False
    count=0
    while  end_flag==False:  #拟合后每列坐标点个数大于行数才进行判断是否存在相交的情况
        #print("第", i+1, "列拟合点坐标个数不等于行数，进行相交矩形检查")
        #print("第", i+1, "列拟合点坐标：", cols_points)
        if count>=20: #若相交矩形检查次数大于10次，则退出循环
            end_flag=True
            break
        count+=1
        for j in range(len(col_points)-1): #遍历该列的拟合点坐标，判断是否存在相交的情况
            point1=col_points[j]  #第一个点
            point2=col_points[j+1]  #第二个点
            f,rect,intersection_area_ratio=merge_rectangles(center_to_rectangle[point1],center_to_rectangle[point2],max_intersection_area_ratio) #判断两个矩形是否有重叠部分，并计算相交面积占比
            if f :#存在相交的情况
                col_inter_flag=True #该列存在相交的情况
                col_points=remove_plots_from_points(col_points,[point1,point2]) #将相交的两个点从该列的拟合点坐标列表中删除
                center_xy=remove_plots_from_points(center_xy,[point1,point2]) #将相交的两个点从中心坐标列表中删除
                merged_center_point=((rect[0][0]+rect[1][0])/2,(rect[0][1]+rect[1][1])/2) #计算合并后的矩形中心坐标
                center_xy.append(merged_center_point) #在中心坐标列表中增加由两相交矩形合并而成的矩形中心坐标
                col_points.append(merged_center_point) #在该列的拟合点坐标列表中增加由两相交矩形合并而成的矩形中心坐标
                col_points=sort_points_by_ymax_to_ymin(col_points) #对该列的拟合点坐标列表按y值从大到小排序
                center_to_rectangle[merged_center_point]=rect #将合并后的矩形的坐标对加入到中心坐标to矩形坐标对的字典中
                rectangle_to_center[rect]=merged_center_point  #将合并后的矩形的中心坐标加入到矩形坐标对to中心坐标的字典中
                rect1=patches.Rectangle(rect[0],rect[1][0]-rect[0][0],rect[1][1]-rect[0][1],linewidth=2,edgecolor='green',facecolor='none') #绘制合并后的矩形
                #ax.add_patch(rect1)
                ax.text(rect[0][0],rect[1][1],str(round(intersection_area_ratio,4)),color='red',fontsize=10) #显示相交面积占比
                if len(col_points)==nums_of_row: #若该列的拟合点坐标列表中点个数等于行数，则退出循环
                    end_flag=True
                    break
            else:
                if j==len(col_points)-2: #若遍历到最后一个点，则退出循环
                    #end_flag=True
                    break
                else:
                    continue
    if col_inter_flag==False and len(col_points)>8: #若该列的拟合点坐标列表中点个数大于行数，但不存在相交的情况，则对相交面积占比大于0.005的矩形进行合并
        rect_for_col_regression=[] #用于存储该列的拟合矩形坐标对
        for point in col_points: #将该列的拟合点坐标列表转为矩形坐标对列表
            rect_for_col_regression.append(center_to_rectangle[point])
        fff,rect1,rect2,s=find_intersect_max_area_rectangles(rect_for_col_regression) #找出两相交矩形面积最大的两个矩形
        if fff and rect1!=None and rect2!=None : #若存在两相交矩形，则进行合并
            point1=rectangle_to_center[rect1]
            point2=rectangle_to_center[rect2]
            #print(point1)
            #print(point2)
            ff,rect,intersection_area_ratio=merge_rectangles(rect1,rect2,0.005) #判断两个矩形是否有重叠部分，并计算相交面积占比
            col_points=remove_plots_from_points(col_points,[point1,point2]) #将相交的两个点从该列的拟合点坐标列表中删除
            center_xy=remove_plots_from_points(center_xy,[point1,point2]) #将相交的两个点从中心坐标列表中删除
            merged_center_point=((rect[0][0]+rect[1][0])/2,(rect[0][1]+rect[1][1])/2) #计算合并后的矩形中心坐标
            col_points.append(merged_center_point) #在该列的拟合点坐标列表中增加由两相交矩形合并而成的矩形中心坐标
            center_xy.append(merged_center_point) #在中心坐标列表中增加由两相交矩形合并而成的矩形中心坐标
            col_points=sort_points_by_ymax_to_ymin(col_points) #对该列的拟合点坐标列表按y值从大到小排序
            center_to_rectangle[merged_center_point]=rect #将合并后的矩形的坐标对加入到中心坐标to矩形坐标对的字典中
            rectangle_to_center[rect]=merged_center_point  #将合并后的矩形的中心坐标加入到矩形坐标对to中心坐标的字典中
            rect1=patches.Rectangle(rect[0],rect[1][0]-rect[0][0],rect[1][1]-rect[0][1],linewidth=2,edgecolor='green',facecolor='none') #绘制合并后的矩形
            #ax.add_patch(rect1)
            ax.text(rect[0][0],rect[1][1],str(round(intersection_area_ratio,4)),color='red',fontsize=10) #显示相交面积占比
            if len(col_points)==nums_of_row: #若该列的拟合点坐标列表中点个数等于行数，则退出循环
                end_flag=True
    return center_xy,col_points



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





#单独拟合最靠近摄像头一列的直线，用以检查第一个点是否缺失
def fit_first_col_line_only(center_xy,ax,position):
    """
    单独拟合最靠近摄像头一列的直线，用以检查第一个点是否缺失
    :param center_xy: 中心点坐标列表
    :param ax: 绘图对象
    :param position: 摄像头位置
    :return:  first_col_num,first_col_line_points_for_regression #返回第一列的点数和拟合点坐标列表 
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
            first_col_line_points_for_regression.append(point) #将该点添加到拟合点坐标列表中
            first_col_num+=1 #则该列的点数加1
    #对拟合点坐标列表按照y坐标进行排序,从大到小)
    first_col_line_points_for_regression=sorted(first_col_line_points_for_regression,key=lambda x:x[1],reverse=True)
    center_xy_cp=remove_plots_from_points(center_xy_cp,first_col_line_points_for_regression) #从中心点坐标列表中删除拟合点坐标列表中的点
    k,b=linear_regression(np.array(first_col_line_points_for_regression)) #对拟合点坐标列表进行直线拟合,得到最靠近摄像头的列的k和b
    #draw_line(k,b,ax,'green',img_size[0],0) #绘制最靠近摄像头的列的直线
    for point in center_xy_cp:
        rectangle=center_to_rectangle[point] #得到该点对应的矩形坐标对
        if is_line_intersect_rectangle(k,b,rectangle): #如果拟合的直线穿过了矩形
            first_col_line_points_for_regression.append(point) #将该点添加到拟合点坐标列表中
            first_col_num+=1 #则该列的点数加1
    #对拟合点坐标列表按照y坐标进行排序,从大到小)
    first_col_line_points_for_regression=set(first_col_line_points_for_regression) #删除重复的点
    first_col_line_points_for_regression=sorted(first_col_line_points_for_regression,key=lambda x:x[1],reverse=True)
    #print("最靠近摄像头的列的拟合点坐标有",len(first_col_line_points_for_regression),"个点,为：",first_col_line_points_for_regression)
    #print("最靠近摄像头的列的点数为：",first_col_num)
    #plt.text(first_col_line_points_for_regression[0][0]+30,first_col_line_points_for_regression[0][1]+30,str(first_col_num),fontsize=12,color='green') #绘制最靠近摄像头的列的点数
    return first_col_num,first_col_line_points_for_regression #返回第一列的点数和拟合点坐标列表 


# 拟合直线
def all_col_lines_regression(center_xy_cp,nums_of_col,position):
    """
    拟合每一列的直线
    :param center_xy_cp: 中心点坐标列表
    :param nums_of_col: 列数
    :return:  points_for_col_regression #返回每一列的拟合点坐标列表 
    """ 
    points_for_col_regression=[]
    for i in range(nums_of_col):  # 遍历每一列
        per_col_num,per_col_line_points_for_regression=fit_first_col_line_only(center_xy_cp,ax,position) # 
        points_for_col_regression.append(per_col_line_points_for_regression) #将每一列的拟合点坐标列表存入points_for_col_regression列表
        center_xy_cp=remove_plots_from_points(center_xy_cp,per_col_line_points_for_regression) #删除拟合点坐标
    points_for_col_regression=sorted(points_for_col_regression,key=lambda x:x[0][0])
    for points in points_for_col_regression:
            k,b=linear_regression(np.array(points)) #调用函数,进行一次线性回归,获得斜率k和截距b
            col_k_list.append(k) #将斜率k存入col_k_list列表
            col_b_list.append(b) #将截距b存入col_b_list列表
    return points_for_col_regression #返回每一列的拟合点坐标列表


def merge_all_col_lines_rectangles(center_xy,points_for_col_regression):
    """
    将每一列的拟合点坐标列表合并到一个列表中，并与矩形坐标对进行合并
    :param center_xy: 中心点坐标列表
    :param points_for_col_regression: 每一列的拟合点坐标列表
    :return:  center_xy ,points_for_col_regression #返回中心点坐标列表和每一列的拟合点坐标列表 
    """
    if len(points_for_col_regression[0])>8: center_xy,points_for_col_regression[0]=merge_col_regression_intersect_rectangles(center_xy,points_for_col_regression[0],ax,position)   #合并第一列直线
    if len(points_for_col_regression[1])>7: center_xy,points_for_col_regression[1]=merge_col_regression_intersect_rectangles(center_xy,points_for_col_regression[1],ax,position)   #合并第二列直线
    if len(points_for_col_regression[2])>7: center_xy,points_for_col_regression[2]=merge_col_regression_intersect_rectangles(center_xy,points_for_col_regression[2],ax,position)   #合并第三列直线
    if len(points_for_col_regression[3])>8: center_xy,points_for_col_regression[3]=merge_col_regression_intersect_rectangles(center_xy,points_for_col_regression[3],ax,position)   #合并第四列直线
    return center_xy,points_for_col_regression



def is_all_col_lines_correct(points_for_col_regression):
    """
    判断每一列的拟合结果是否正确
    :param points_for_col_regression: 每一列的拟合点坐标列表
    :return:  four_cols_flag #返回四列标志位
    """
    four_cols_flag=True #设置四列标志位
    if len(points_for_col_regression[0])!=8: #如果第一列点数不等于8
        four_cols_flag=False #设置四列标志位为False
    if len(points_for_col_regression[1])!=7: #如果第二列点数不等于7
        four_cols_flag=False #设置四列标志位为False
    if len(points_for_col_regression[2])!=7: #如果第三列点数不等于7
        four_cols_flag=False #设置四列标志位为False
    if len(points_for_col_regression[3])!=8: #如果第四列点数不等于8
        four_cols_flag=False #设置四列标志位为False
    print("四列标志位为：",four_cols_flag) #打印输出四列标志位
    return four_cols_flag #返回四列标志位



def get_row_info(points_for_col_regression):
    """
    得到每一行的点数和拟合点坐标列表
    :param points_for_col_regression: 每一列的拟合点坐标列表
    :return:  points_for_row_regression #返回每一行的拟合点坐标列表 
    """
    points_for_row_regression=[[],[],[],[],[],[],[],[]] #定义一个列表,用于存储第一行的点坐标列表,第二行的点坐标列表,第三行的点坐标列表,第四行的点坐标列表,第五行的点坐标列表,第六行的点坐标列表,第七行的点坐标列表,第八行的点坐标列表
    for i in range(8): #遍历8行
        for j in range(4): #遍历四列
            #k,b=linear_regression(np.array(points_for_col_regression[j])) #调用函数,进行一次线性回归,获得斜率k和截距b
            #col_k_list.append(k) #将斜率k存入col_k_list列表
            #col_b_list.append(b) #将截距b存入col_b_list列表
            if i==0: #如果是第一行
                if j==0 or j==3: #如果是第一列或第四列
                    points_for_row_regression[i].append(points_for_col_regression[j][i]) #
                    point=points_for_col_regression[j][i] #
                    ax.text(point[0], point[1], str('('+str(i)+','+str(j+1)+')'), color='black', fontsize=10)  # 绘制中心坐标对应的行列号
            else: #如果不是第一行
                    if j==0 or j==3: #如果是第一列或第四列
                        points_for_row_regression[i].append(points_for_col_regression[j][i]) #
                        point=points_for_col_regression[j][i] #
                        ax.text(point[0], point[1], str('('+str(i)+','+str(j+1)+')'), color='black', fontsize=10)  # 绘制中心坐标对应的行列号
                    else: #如果不是第一列或第四列
                        points_for_row_regression[i].append(points_for_col_regression[j][i-1]) #
                        point=points_for_col_regression[j][i-1] #
                        ax.text(point[0], point[1], str('('+str(i)+','+str(j+1)+')'), color='black', fontsize=10)  # 绘制中心坐标对应的行列号
        print("第",i+1,"行点坐标有：",len(points_for_row_regression[i]),"个点,为：",points_for_row_regression[i]) #打印输出第i+1行点坐标个数
        k,b=linear_regression(np.array(points_for_row_regression[i])) #调用函数,进行一次线性回归,获得斜率k和截距b
        row_k_list.append(k) #将斜率k存入row_k_list列表
        row_b_list.append(b) #将截距b存入row_b_list列表
    return points_for_row_regression #返回每一行的拟合点坐标列表





def missing_col_lines_points():
    """
    补充缺失的点坐标
    :return:  None
    """
#判断是否存在缺失的点,如果存在缺失的点,则尝试进行补充
    for i in range(8): #遍历后7行
        if i==0: #如果是第一行
            continue #跳过第一行
        maybe_row_points=[] #定义一个列表,用于存储可能的行坐标列表
        true_row_points=[] #定义一个列表,用于存储真实的行坐标列表
        for j in range(4): #遍历四列
            if j==0 or j==3: #如果是第一列或第四列
                if len(points_for_col_regression[j])==8: #如果第一列或第四列点数为8
                    true_row_points.append(points_for_col_regression[j][i]) #将真实的行坐标添加到列表中
                    maybe_row_points.append(points_for_col_regression[j][i]) #将可能的行坐标添加到列表中
                if len(points_for_col_regression[j])<8: #如果第一列或第四列点数小于8
                    if len(points_for_col_regression[j])>i: 
                        maybe_row_points.append(points_for_col_regression[j][i]) #将可能的行坐标添加到列表中
                    else: 
                        maybe_row_points.append(points_for_col_regression[j][0]) #将可能的行坐标添加到列表中
            else: #如果不是第一列或第四列
                if len(points_for_col_regression[j])==7: #如果不是第一列或第四列点数为7
                    true_row_points.append(points_for_col_regression[j][i-1]) #将真实的行坐标添加到列表中
                    maybe_row_points.append(points_for_col_regression[j][i-1]) #将可能的行坐标添加到列表中
                if len(points_for_col_regression[j])<7: #如果不是第一列或第四列点数小于7
                    if len(points_for_col_regression[j])>i: 
                        maybe_row_points.append(points_for_col_regression[j][i-1]) #将可能的行坐标添加到列表中
                    else: 
                        maybe_row_points.append(points_for_col_regression[j][0]) #将可能的行坐标添加到列表中
        #print("第",i,"行可能的点坐标有：",len(maybe_row_points),"个点,为：",maybe_row_points) #打印输出第i+1行可能的点坐标个数
        #print("第",i,"行真实的点坐标有：",len(true_row_points),"个点,为：",true_row_points) #打印输出第i+1行真实的点坐标个数
        row_k,row_b=linear_regression(np.array(true_row_points)) #调用函数,进行一次线性回归,获得斜率k和截距b
        #draw_line(row_k,row_b,ax,'orange',0,img_size[0])
        for point in maybe_row_points:  # 遍历可能的点坐标
                        if is_line_intersect_rectangle(row_k,row_b,center_to_rectangle[point]):  # 判断该行直线是否穿过可能的点的矩形框
                            #print("可能的点坐标：",point)
                            continue
                        else:  # 如果该行直线不穿过该点的矩形框,则认为该点是错误的
                            #print("错误的点坐标：",point)
                            for k in range(len(points_for_col_regression)) :  # 遍历每列，找到错误的点属于哪一列
                                if point in points_for_col_regression[k]:  # 如果该点属于该列
                                    #print("该点属于第",k+1,"列")
                                    if len(points_for_col_regression[k])+1>nums_of_row:  # 如果该列的点数等于行数+1,说明该列的拟合结果不正确,尝试进行合并矩形
                                        points_for_col_regression[k]=merge_col_regression_intersect_rectangles(center_xy,points_for_col_regression[k],ax,position)  # 合并该列的点坐标
                                    col_k,col_b=linear_regression(np.array(points_for_col_regression[k]))  # 计算该列直线,计算两条直线的交点
                                    missing_point_x=int(abs((-col_b+row_b)/(col_k-row_k)))  # 计算缺失的点的横坐标
                                    missing_point_y=int(abs(col_k*missing_point_x+col_b))  # 计算缺失的点的纵坐标
                                    missing_point=(missing_point_x,missing_point_y)  # 缺失的点的坐标
                                    #sample_points=points_for_col_regression[k]  # 取该列的点坐标作为参考点
                                    #print("缺失的点坐标：",missing_point,"属于第",k+1,"列 第",i+1,"行")
                                    if i>=len(points_for_col_regression[k]):  # 如果该行的序号大于等于该列的点数,说明缺失的点在最后一个
                                        points_for_col_regression[k].append(missing_point)
                                    else:
                                        points_for_col_regression[k]=insert_point_to_list(missing_point,points_for_col_regression[k],i)  # 插入缺失的点到该列的点列表中
                                    #print("该列的点坐标：",points_for_col_regression[k])
                                    center_xy.append(missing_point) #将缺失点加入中心点坐标列表中
                                    sample_points=true_row_points  # 取真实的行坐标作为参考点
                                    coordinate=get_fill_rectangle_data(missing_point,sample_points) #计算补充矩形的坐标对数据
                                    center_to_rectangle[missing_point]=coordinate #将缺失点的坐标对数据加入中心点坐标对字典中
                                    rectangle_to_center[coordinate]=missing_point #将缺失点的坐标对数据加入矩形坐标对字典中
                                    coordinates.append(coordinate) #将缺失点的坐标对数据加入坐标对列表中













if __name__ == '__main__':
    file_path__="../col_and_row_regression/test-data/labels/vacant2_446.txt"
    save_path__='../col_and_row_regression/test_result'
    all_img_flag__=False #是否处理所有图片 
    init_global_var(all_img_flag_in=all_img_flag__,file_path_in=file_path__,save_path_in=save_path__,count=1)
    center_xy,center_xy_array,center_xy_nums=calculate_center_xy(coordinates)  #调用函数,计算中心点坐标,中心点坐标数组,中心点坐标个数
    center_xy_cp=center_xy.copy() #  copy center_xy列表
    position=judge_camera_position(center_xy,5) #判断摄像头位置
    first_point=get_first_point(center_xy) #获取第一个点(最靠近摄像头的点)
    #plt.plot(first_point[0],first_point[1],marker='o',markersize=12,color='green') #绘制第一个点
    is_first_point_correct_flag=is_first_point_correct(first_point,center_xy,position) #检查第一个点是否正确
    #plt.text(first_point[0]+30,first_point[1]+30,is_first_point_correct_flag,fontsize=12,color='green') #绘制第一个点的序号
    print("摄像头位置为：",position) #打印输出摄像头位置
    print("中心点坐标个数为：",center_xy_nums) #打印输出中心点坐标个数
    print("第一个点为：" ,first_point) #打印输出第一个点
    print("第一个点是否正确：" ,is_first_point_correct_flag)
    #1.如果中心点坐标数小于30-10,则认为摄像头位置不正确,退出程序
    if center_xy_nums<30-10: 
        #print("中心点个数为：",len(center_xy)) #打印输出中心点坐标个数
        draw_center_point_and_rect(center_xy,ax,img_size) #绘制中心点和矩形框
        set_axis_and_save(0,ax,img_name,img_size,save_dir,save_flag)#设置坐标轴范围和保存路径等
        put_text_of_camera_position(position,center_xy,ax,5) #绘制摄像头位置
        exit()
    points_for_col_regression=all_col_lines_regression(center_xy_cp,nums_of_col,position) #调用函数,拟合每一列的直线
    center_xy,points_for_col_regression=merge_all_col_lines_rectangles(center_xy,points_for_col_regression) #调用函数,合并重合矩形
    for i in range(4): #遍历四列
        print("第",i+1,"列点坐标有：",len(points_for_col_regression[i]),"个点,为：",points_for_col_regression[i]) #打印输出第i+1列点坐标个数
    four_cols_flag=is_all_col_lines_correct(points_for_col_regression) #调用函数,判断每一列的拟合结果是否正确
    if four_cols_flag==True: #如果四列标志位为True
        points_for_row_regression=get_row_info(points_for_col_regression) #调用函数,得到每一行的点数和拟合点坐标列表
        draw_col_line(col_k_list,col_b_list,points_for_col_regression,ax,img_size) #绘制四列直线
        draw_row_line(row_k_list,row_b_list,points_for_row_regression,ax,img_size) #绘制八行直线
        draw_center_point_and_rect(center_xy,ax,img_size) #绘制中心点和矩形框
        put_text_of_camera_position(position,center_xy,ax,5) #绘制摄像头位置
        set_axis_and_save(1,ax,img_name,img_size,save_dir,save_flag)#设置坐标轴范围和保存路径
        exit()
    if four_cols_flag==False: #如果四列标志位为False
        missing_col_lines_points() #判断是否存在缺失的点,如果存在缺失的点,则尝试进行补充
        center_xy,points_for_col_regression=merge_all_col_lines_rectangles(center_xy,points_for_col_regression) #调用函数,合并重合矩形
        four_cols_flag=is_all_col_lines_correct(points_for_col_regression) #调用函数,判断每一列的拟合结果是否正确
        if four_cols_flag==True: #如果四列标志位为True
            points_for_row_regression=get_row_info(points_for_col_regression) #调用函数,得到每一行的点数和拟合点坐标列表
            draw_col_line(col_k_list,col_b_list,points_for_col_regression,ax,img_size) #绘制四列直线
            draw_row_line(row_k_list,row_b_list,points_for_row_regression,ax,img_size) #绘制八行直线
            draw_center_point_and_rect(center_xy,ax,img_size) #绘制中心点和矩形框
            put_text_of_camera_position(position,center_xy,ax,5) #绘制摄像头位置
            set_axis_and_save(1,ax,img_name,img_size,save_dir,save_flag)#设置坐标轴范围和保存路径
            exit()
        else: #如果四列标志位为False
            #draw_col_line(col_k_list,col_b_list,points_for_col_regression,ax,img_size) #绘制四列直线
            draw_center_point_and_rect(center_xy,ax,img_size) #绘制中心点和矩形框
            put_text_of_camera_position(position,center_xy,ax,5) #绘制摄像头位置
            set_axis_and_save(0,ax,img_name,img_size,save_dir,save_flag)#设置坐标轴范围和保存路径
            exit()