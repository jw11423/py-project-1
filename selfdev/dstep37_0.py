if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../selfpkg'))

import numpy as np
from dezero import Variable
import dezero.functions as F
from matplotlib import pyplot as plt

# x = Variable(np.linspace(-2, 2, 200))
x = Variable(np.array(1.0))
y = F.sin(x)

print(y)

# plt.plot(x.data, y.data, '--', color='blue')
# plt.savefig('test37_01.png')

x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.sin(x)

print(y)
plt.plot(x.data, y.data, '--', color='blue')
plt.savefig('test37_01.png')


x = Variable(np.array([[1,2,3],[4,5,6]]))
c = Variable(np.array([[10,20,30],[40,50,60]]))
y = x + c

print(y)

x = Variable(np.array([[1,2,3],[4,5,6]]))
c = Variable(np.array([[10,20,30],[40,50,60]]))
y = x * c

print(y)

x = Variable(np.array([[1,2,3],[4,5,6],[7,8,9]]))
c = Variable(np.array([[10,20,30],[40,50,60],[70,80,90]]))
y = x * c

print(y)


# ######################################################################################################
# import matplotlib.pyplot as plt
# ## 라인 종류 - 이름으로 지정하기
 
# line_type = {'solid line' : 'solid', ## 라인 유형
#              'dashed line' : 'dashed',
#              'dash-dotted line' : 'dashdot',
#              'dotted line' : 'dotted'}
 
# fig = plt.figure(figsize=(10,5)) ## 캔버스 생성
# fig.set_facecolor('white') ## 캔버스 색상 하얀색으로 설정
 
# yticks_coord = range(len(line_type)) ## y축 눈금 좌표
# yticks_label = list(line_type.keys()) ## y축 눈금 라벨
 
# plt.xticks([]) ## x축 제거
# plt.yticks(yticks_coord, yticks_label) ## y축 생성
 
# for i in yticks_coord:
#     plt.hlines(y=i,xmin=0,xmax=10,linestyle=line_type[yticks_label[i]]) ## 수평 직선 생성
# plt.savefig('test37_02.png')

# ######################################################################################################
# import matplotlib.pyplot as plt
# ## 라인 종류 - 이름으로 지정하기
 
# line_type = {'-' : '-', ## 라인 유형
#              '--' : '--',
#              '-.' : '-.',
#              ':' : ':'}
 
# fig = plt.figure(figsize=(10,5)) ## 캔버스 생성
# fig.set_facecolor('white') ## 캔버스 색상 하얀색으로 설정
 
# yticks_coord = range(len(line_type)) ## y축 눈금 좌표
# yticks_label = list(line_type.keys()) ## y축 눈금 라벨
 
# plt.xticks([]) ## x축 제거
# plt.yticks(yticks_coord, yticks_label, fontsize=15) ## y축 생성
 
# for i in yticks_coord:
#     plt.hlines(y=i,xmin=0,xmax=10,linestyle=line_type[yticks_label[i]]) ## 수평 직선 생성
# plt.savefig('test37_03.png')

# ######################################################################################################
# import matplotlib.pyplot as plt
# ## 라인 종류 - 튜플
# ## 라인 유형 설정
# line_type_value = [(0,(1,0)), (0,(4.5,1.5)), (0,(3,1,1,1)), (0,(4,2,2,2,1,2))]
# line_type_key = [str(x) for x in line_type_value]
# line_type = dict(zip(line_type_key,line_type_value))
 
# fig = plt.figure(figsize=(12,5)) ## 캔버스 생성
# fig.set_facecolor('white') ## 캔버스 색상 하얀색으로 설정
 
# yticks_coord = range(len(line_type)) ## y축 눈금 좌표
# yticks_label = list(line_type.keys()) ## y축 눈금 라벨
 
# plt.xticks([]) ## x축 제거
# plt.yticks(yticks_coord, yticks_label) ## y축 생성
 
# for i in yticks_coord:
#     plt.hlines(y=i,xmin=0,xmax=10,linestyle=line_type[yticks_label[i]]) ## 수평 직선 생성

# # plt.subplots_adjust(wspace=0.35, hspace=0.5)
# plt.savefig('test37_04.png')
# ######################################################################################################
# import matplotlib.pyplot as plt
# ## 라인 종류 - 오프셋 변화
# ## 라인 유형 설정
# line_type_value = [(5,(25,10)), (10,(25,10)), (15,(25,10)), (20,(25,10))]
# line_type_key = [str(x) for x in line_type_value]
# line_type = dict(zip(line_type_key,line_type_value))
 
# fig = plt.figure(figsize=(10,5)) ## 캔버스 생성
# fig.set_facecolor('white') ## 캔버스 색상 하얀색으로 설정
 
# yticks_coord = range(len(line_type)) ## y축 눈금 좌표
# yticks_label = list(line_type.keys()) ## y축 눈금 라벨
 
# plt.xticks([]) ## x축 제거
# plt.yticks(yticks_coord, yticks_label) ## y축 생성
 
# for i in yticks_coord:
#     plt.hlines(y=i,xmin=0,xmax=10,linestyle=line_type[yticks_label[i]]) ## 수평 직선 생성
# plt.savefig('test37_05.png')
# ######################################################################################################
# import matplotlib.pyplot as plt
# ## 적용 - 선 그래프
# ## 데이터 만들기
# weekday = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
# visits = [10,15,9,7,3,10,20]
 
# fig = plt.figure(figsize=(10,10)) ## 캔버스 생성
# fig.set_facecolor('white') ## 캔버스 색상 하얀색으로 설정
 
# xticks_coord = range(len(weekday)) ## x축 눈금 좌표
 
# plt.plot(xticks_coord,visits,linestyle=(0,(4,2,2,2))) ## 선 그래프 생성
# plt.xticks(xticks_coord,weekday,fontsize=15) ## x축 눈금 생성
# plt.yticks([0,5,10,15,20]) ## y축 눈금 생성, 화면에 표시될 눈금값은 0, 5, 10, 15, 20
# plt.savefig('test37_06.png')
# ######################################################################################################
# import matplotlib.pyplot as plt
# ## 응용 - 바 차트
# ## 데이터 만들기
# weekday = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
# visits = [10,15,9,7,3,10,20]
 
# fig = plt.figure(figsize=(10,10)) ## 캔버스 생성
# fig.set_facecolor('white') ## 캔버스 색상 하얀색으로 설정
 
# xticks_coord = range(len(weekday)) ## x축 눈금 좌표
 
# plt.bar(xticks_coord,visits,edgecolor = 'k', linestyle='dashed', linewidth=2) ## 선 그래프 생성
# plt.xticks(xticks_coord,weekday,fontsize=15) ## x축 눈금 생성
# plt.yticks([0,5,10,15,20]) ## y축 눈금 생성, 화면에 표시될 눈금값은 0, 5, 10, 15, 20
# plt.savefig('test37_07.png')
# ######################################################################################################

# # 그림이 생성 된 후 Matplotlib 에서 그림 크기를 변경하려면 set_size_inches
# # from matplotlib import pyplot as plt

# # fig1 = plt.figure(1)
# # plt.plot([[1,2], [3, 4]])
# # fig2 = plt.figure(2)
# # plt.plot([[1,2], [3, 4]])

# # fig1.set_size_inches(3, 3)
# # fig2.set_size_inches(4, 4)

# # plt.savefig('test37_05.png')