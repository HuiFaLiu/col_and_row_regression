import os
import subprocess
import time
import subprocess

def get_txt_file_paths(txt_dir):  #获得txt文件路径
    txt_file_paths = []
    for root, dirs, files in os.walk(txt_dir):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                txt_file_paths.append(file_path)
    return txt_file_paths



txt_dir = "../col_and_row_regression/test-data/labels"  #所有文件夹路径,里面的文件是，文件名.txt，包含样本信息
all_path="../col_and_row_regression/test-data/labels" 
maybe_right_paths_txt = "../col_and_row_regression/maybe_right_paths.txt"
test_paths_txt='../col_and_row_regression/test_paths.txt'  #存储测试样本路径的txt文件
test_for_4cols_8778_txt='../col_and_row_regression/test_for_4cols_8778.txt'  #存储四列布局的测试样本路径的txt文件
txt_file_paths = get_txt_file_paths(txt_dir)  #获得样本文本文件路径  
maybe_paths=[]  #存储可能正确的路径
test_paths=[]  #存储测试路径
test_for_4cols_8778_paths=[]  #存储四列布局的测试路径

# 读取可能正确的路径
with open(maybe_right_paths_txt, 'r') as f:
    for line in f.readlines():
        line = line.strip()
        if line.endswith(".txt"):
            maybe_paths.append(line)

# 读取测试路径
with open(test_paths_txt, 'r') as f:
    for line in f.readlines():
        line = line.strip()
        if line.endswith(".txt"):
            test_paths.append(line)


# 读取四列布局的测试路径
with open(test_for_4cols_8778_txt, 'r') as f:
    for line in f.readlines():
        line = line.strip()
        if line.endswith(".txt"):
            test_for_4cols_8778_paths.append(line)


i=1  #计数器
start_time = time.time()  #开始计时
k=3  #选择运行模式，1：遍历所有txt文件，2：遍历可能的正确路径，3：遍历测试路径，4：遍历四列布局的路径

if k==1:
    # 遍历每个文件路径并执行
    for txt_file_path in txt_file_paths:
        # 运行命令行命令
        print("第",i,"个文件路径：",txt_file_path)
        subprocess.call("python  ../col_and_row_regression/code/regression_for_5cols.py " + txt_file_path, shell=True)
        i+=1

if k==2:
    # 遍历可能的正确路径并执行
    for maybe_path in maybe_paths:
        # 运行命令行命令
        print("第",i,"个可能的正确路径：",maybe_path)
        subprocess.call("python  ../col_and_row_regression/code/regression_for_5cols.py " + maybe_path, shell=True)
        i+=1

if k==3:
    # 遍历测试路径并执行
    for test_path in test_paths:
        # 运行命令行命令
        print("第",i,"个测试路径：",test_path)
        subprocess.call("python  ../col_and_row_regression/code/regression_for_5cols.py " + test_path, shell=True)
        i+=1

if k==4:
    # 遍历四列布局的路径并执行
    for test_for_4cols_8778_path in test_for_4cols_8778_paths:
        print("第",i,"个四列布局的测试路径：",test_for_4cols_8778_path)
        subprocess.call("python  ../col_and_row_regression/code/regression_for_4cols.py " + test_for_4cols_8778_path, shell=True)
        i+=1

#计算总用时
end_time = time.time()
print("总用时：", end_time - start_time)

