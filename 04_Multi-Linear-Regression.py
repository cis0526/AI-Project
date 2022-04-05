import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

plt.rc('font', family='Malgun Gothic')

#공부시간 X와 성적 Y의 리스트를 만듭니다.
data = [[2, 0, 81], [4, 4, 93], [6, 2, 91], [8, 3, 97]]
x1 = [i[0] for i in data]
x2 = [i[1] for i in data]
y = [i[2] for i in data]

#그래프로 확인해 봅니다.
#plt.figure(figsize=(10,10))
ax = plt.axes(projection='3d') #3차원 그래프
ax.set_xlabel('공부시간')
ax.set_ylabel('과외공부')
ax.set_zlabel('점수')
ax.dist = 11 #ax.dist = 10작을수록 큐브가 더 가깝게 나타나고 값이 클수록 더 멀리 보입니다.
ax.scatter(x1, x2, y)
plt.show()

#리스트로 되어 있는 x와 y값을 넘파이 배열로 바꾸어 줍니다.(인덱스를 주어 하나씩 불러와 계산이 가능해 지도록 하기 위함입니다.)
x1_data = np.array(x1)
x2_data = np.array(x2)
y_data = np.array(y)

# 기울기 a와 절편 b의 값을 초기화 합니다.
a1 = 0
a2 = 0
b = 0

#학습률을 정합니다.
lr = 0.02

#몇 번 반복될지를 설정합니다.(0부터 세므로 원하는 반복 횟수에 +1을 해 주어야 합니다.)
epochs = 2001

#경사 하강법을 시작합니다.
'''
for i in range(epochs): # epoch 수 만큼 반복
    y_pred = a1 * x1_data + a2 * x2_data + b  #y를 구하는 식을 세웁니다
    error = y_data - y_pred  #오차를 구하는 식입니다.
    a1_diff = -(2/len(x1_data)) * sum(x1_data * (error)) # 오차함수를 a1로 미분한 값입니다.
    a2_diff = -(2/len(x2_data)) * sum(x2_data * (error)) # 오차함수를 a2로 미분한 값입니다.
    b_new = -(2/len(x1_data)) * sum(y_data - y_pred)  # 오차함수를 b로 미분한 값입니다.
    a1 = a1 - lr * a1_diff  # 학습률을 곱해 기존의 a1값을 업데이트합니다.
    a2 = a2 - lr * a2_diff  # 학습률을 곱해 기존의 a2값을 업데이트합니다.
    b = b - lr * b_new  # 학습률을 곱해 기존의 b값을 업데이트합니다.
    if i % 100 == 0:    # 100번 반복될 때마다 현재의 a1, a2, b값을 출력합니다.
        print("epoch=%.f, 기울기1=%.04f, 기울기2=%.04f, 절편=%.04f" % (i, a1, a2, b))

#참고 자료, 다중 선형회귀 '예측 평면' 3D로 보기
'''
import statsmodels.api as statm  #statsmodels R에서 구현하고 있는 내용을 그대로 구현하고자 하는 목적으로 만들어진 패키지

import statsmodels.formula.api as statfa
#from matplotlib.pyplot import figure

X = [i[0:2] for i in data]
y = [i[2] for i in data]

'''
X_1=statm.add_constant(X) #상수항 결합
#print(X_1)
results=statm.OLS(y,X_1).fit() # 잔차제곱합을 최소화하는 가중치 벡터를 구하는 방법.
                             # fit() 최적의 값, 객체값 반환
print(results.params)
#[77.85714286  1.5   2.28571429]  [절편, 기울기1, 기울기2]
'''

hour_class=pd.DataFrame(X,columns=['study_hours','private_class'])
#print(hour_class)

hour_class['Score']=pd.Series(y)
#print(hour_class)
model = statfa.ols(formula='Score ~ study_hours + private_class', data=hour_class)
    #formula: 모델을 지정하는 공식입니다.
# formula 문자열을 만드는 방법은 ~ 기호의 왼쪽에 종속변수의 이름을 넣고 ~ 기호의 오른쪽에 독립변수의 이름을 넣는다. 만약 독립변수가 여러개일 경우에는 patsy 패키지의 formula 문자열을 만드는 법을 따른다.
# data : 모델에대한 데이터 array
#print(model)
results_formula = model.fit()
print(results_formula.params)

a, b = np.meshgrid(np.linspace(hour_class.study_hours.min(),hour_class.study_hours.max(),100),
                   np.linspace(hour_class.private_class.min(),hour_class.private_class.max(),100))

X_ax = pd.DataFrame({'study_hours': a.ravel(), 'private_class': b.ravel()})
fittedY=results_formula.predict(exog=X_ax)

fig = plt.figure()
graph = fig.add_subplot(111, projection='3d')

graph.scatter(hour_class['study_hours'],hour_class['private_class'],hour_class['Score'],
              c='blue',marker='o', alpha=1)
graph.plot_surface(a,b,fittedY.values.reshape(a.shape),
                   rstride=1, cstride=1, color='none', alpha=0.4)
graph.set_xlabel('study hours')
graph.set_ylabel('private class')
graph.set_zlabel('Score')
graph.dist = 11

plt.show()