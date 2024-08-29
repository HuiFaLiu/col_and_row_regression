 # Col and Row Regression
1. Introduction 
    - 目录结构
    project_root/  # 项目根目录
    │
    ├── col_and_row_regression/  # 项目目录
    │   ├── code/  # 代码目录
    │   │   ├── test_all_image.py  # 测试所有图片
    |   |   |── regression_for_4cols.py  # 处理4列图片主程序
    |   |   |── regression_for_5cols.py  # 处理5列图片主程序
    │   ├── test_data/  # 测试数据集，包括图片和数据txt文件
    │   │   ├──labels/  # 标签文件，每个文件对应一张图片的矩形框数据，最后一行是图片大小
    │   │   │   ├──vacant_13.txt
    │   │   |   ├──......
    │   │   ├──vacant_13.jpg  # 图片文件
    │   |   ├──.....
    │   ├── test_result/  # 测试结果，包括两个txt文件，分别保存拟合正确的终端输出和拟合错误的终端输出，以及拟合正确的图片和拟合错误的图片
    │   │   ├──.....
    │   ├── test_paths.txt  # 存储测试样本路径的txt文件(5列)
    │   ├── maybe_right_paths.txt    # 存储可能正确的样本路径的txt文件(5列)
    │   ├── test_for_4cols_8778.txt  # 4列图片测试样本路径(8778)
    │   ├── test_for_4cols_all.txt    # 4列图片测试样本路径(所有)
    │   ├── requirements.txt  # 依赖库列表
    │   ├── README.md    # 说明文档

2. Col and Row Regression
    - 功能介绍
        - 该项目主要是对矩形框数据进行直线拟合，对矩形框的中心坐标进行行与列的编号
        - 该项目主要使用了OpenCV库，matplotlib库，numpy库，以及其他一些常用的库
    - 运行方式
        - 运行环境：windows 11，python3.9.19，opencv-python库，matplotlib库，numpy库等
        - 运行步骤：
            1. 下载项目包，解压到指定目录
            2. 安装依赖库：pip install -r requirements.txt  #(可能依赖库过多，建议先运行一次代码，查看报错信息，根据报错信息安装缺失的库)
            3. 按需要更改 redirect_output_to_files("../col_and_row_regression/test_result/normal_output.txt", "../col_and_row_regression/test_result/error_output.txt")中的路径，将终端输出重定向到指定文件
            4. 按需要更改save_dir = "../col_and_row_regression/test_result"   # 将测试结果保存到指定目录
            5. 如果只需要对一张样本进行测试，先将regression_for_4cols.py和regression_for_5cols.py中的all_img_flag设置为False，然后再修改图像路径，如：
                # 读取图像路径
                img_path="../col_and_row_regression/test-data/labels/vacant2_825.txt"  # 样本路径，可以替换为其他路径
                然后运行本程序，即可对单张样本进行测试，结果将直接显示在终端和matplotlib窗口中。
            6. 如果需要对所有样本进行测试，先将regression_for_4cols.py和regression_for_5cols.py中的all_img_flag设置为True，然后再根据需要修改test_all_image.py中的测试
            模式：k=2  #选择运行模式，1：遍历所有txt文件(4列或5列)，2：遍历可能的正确路径(5列)，3：遍历测试路径(5列) ，4：遍历测试路径(4列)
            再运行test_all_image.py，即可对所有样本进行测试，终端的输出将保存到指定目录(test_result文件夹)的normal_output.txt和error_output.txt文件中，绘制的图片将保存到指定目录的test_result文件夹中。

3. Algorithm Principles 
    - 算法原理(以五列的为例)
        - 3.1.读取矩形框数据，包括中心坐标，宽高，以及图片大小(img_size)，计算得矩形框的左上角坐标和右下角坐标的坐标对数据列表coordinates
        - 3.2.根据坐标对数据，计算得到矩形中心坐标数据列表center_xy
        - 3.3.根据中心坐标列表center_xy，判断摄像头位置position，进而获得最靠近摄像头位置的第一个点first_point
        - 3.4.依据position获得first_point在center_xy列表中的排序,进而判断第一个点是否正确，得到标志位is_first_point_correct_flag
        - 3.5.拟合得到最靠近摄像头一列的点数first_col_num
        - 3.6.进行初次的列拟合，获得center_xy_cp1 , numbers_of_col, numbers_of_row,position,points_for_col_regression_cp1数据
        - 3.7.计算所有列的拟合结果的分数sum(col_flag_scores)，并对全局变量进行重新初始化
        - 3.8.根据现有信息判断能否正确进行列拟合
            - 3.8.1.如果中心点坐标数小于30-10,则认为摄像头位置不正确,退出程序
            - 3.8.2.如果如果第一个点正确,且第一列点数大于等于行数6,且中心点坐标个数大于等于30,拟合直线列数为5,且所有列的拟合结果正确,则认为所有行列拟合结果可能正确，进行正式的列拟合
                - 3.8.2.1.调用函数get_col_info(coordinates,ax,img_size)，获得center_xy,nums_of_col,nums_of_row,position,points_for_col_regression数据，依据sum(all_col_flags)是否等于5，判断所有列拟合是否正确
                    - 3.8.2.1.1.如果sum(all_col_flags)等于5，则认为所有列拟合结果正确，则获取行拟合点的坐标，调用get_row_and_col_matrix函数获得列直线，行直线的参数以及行列矩阵，并在图片中绘制相关信息，保存到指定目录的test_result文件夹中，退出程序
                    - 3.8.2.1.2.如果所有行列拟合结果不正确,则退出程序
            - 3.8.3.如果第一个点不正确,或第一列点数小于6,或中心点坐标个数大于等于30-10,或所有列的拟合结果不正确，则认为所有行列拟合结果可能不正确，可能需要进行补全的列拟合
                - 3.8.3.1.调用函数get_col_info(coordinates,ax,img_size)，获得center_xy,nums_of_col,nums_of_row,position,points_for_col_regression数据，遍历每一行，通过判断正确的行直线是否通过可能错误的点对应的矩形，进而判断该点是否真的错误，在哪行哪列缺失点，将该行正确点拟合的行直线与该列所有点拟合的列直线的交点作为缺失点的坐标，取该列的点作为样本点，进而获得该缺失点对应的矩形框的宽高数据，进而获得矩形框坐标对数据，把缺失点数据加入到center_xy,points_for_col_regression,coordinates,center_to_rectangle,rectangle_to_center等数据中；检查每列直线拟合结果是否正确,合并错误的重合矩形，按横坐标排序每列的点y坐标,这里应该是按纵坐标从大到小排序,利用sum(all_col_line_flags)判断所有列直线拟合是否正确
                    - 3.8.3.1.1.如果sum(all_col_line_flags)==5，说明所有列拟合都成功，则获取行拟合点的坐标，调用get_row_and_col_matrix函数获得列直线，行直线的参数以及行列矩阵，并在图片中绘制相关信息，保存到指定目录的test_result文件夹中，退出程序
                    - 3.8.3.1.2.如果sum(all_col_line_flags)<5，存在有列拟合错误的情况，重新获取矩形坐标对数据，coordinates,img_size=get_rectangle_data.read_rectangles_from_txt(img_path) ，对全局变量进行重新初始化，main_process() #调用主函数(主要解决列拟合不太准确的样本)

4. Conclusion
    - 本算法目前的(5列)有效样本有458个，目前可以正确拟合的样本有445个，准确率为97.16%，可以作为初步的行列编号算法；
    - 本算法目前的(4列)样本有7个(包含2个无效样本)，目前可以正确拟合的样本有4个，准确率为80%，还需要更多的样本进行测试迭代
    - 算法边界
        - 目前算法能够正确拟合的矩形框数据，行数为6行，列数为5列，如果是4列，行数为8 7 7 8的布局可以拟合大部分，其他四列布局的暂时无法拟合。
        - 目前算法能够正确合并的矩形框数据，需满足某列点数大于6，且错误的点的矩形框与其他点的矩形框存在较大重合(且最好是在靠近摄像头位置的部分)
        - 目前算法可以补全的矩形框数据，需满足至少有两列点数大于等于6，且每行缺失的点数少于等于4，且正确行的点排列较为整齐，在一条直线上，该直线都通过了这些点对应的矩形框



注意：col_and_row_regression文件夹下的:
1. imgs_for_4cols文件夹存放的是4列图片，可以删去
2. test_result(四列8778)文件夹存放的是4列图片的测试结果，可以删去
3. test_result(五列)文件夹存放的是5列图片的测试结果(版本v1)，可以删去
4. test_result(五列v2)文件夹存放的是5列图片的测试结果(版本v2)，可以删去
5. 情况分类.docx是对测试样本的简单分类，
