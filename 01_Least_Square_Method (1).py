import numpy as np
import matplotlib.pyplot as plt

# x 값과 y값
x=[2, 4, 6, 8]
y=[81, 93, 91, 97]

# x와 y의 평균값
mx = np.mean(x)
my = np.mean(y)
print("x의 평균값:", mx)
print("y의 평균값:", my)

# 기울기 공식의 분모
divisor = sum([(mx - i)**2 for i in x])

# 기울기 공식의 분자
def top(x, mx, y, my):

    d = 0
    for i in range(len(x)):
        d += (x[i] - mx) * (y[i] - my)
    return d
'''
    d= sum([(x[i] - mx) * (y[i] - my) for i in range(len(x))]) 
    return d
'''
dividend = top(x, mx, y, my)

print("분모:", divisor)
print("분자:", dividend)

# 기울기와 y 절편 구하기
a = dividend / divisor
b = my - (mx*a)

# 출력으로 확인
print("기울기 a =", a)
print("y 절편 b =", b)


x_data = np.array(x)
y_data = np.array(y)

y_pred = a * x_data + b #예측 값
#print(x_data)
#print(y_pred)

#그래프로 나타내 봅니다.
plt.rc('font', family='Malgun Gothic')
plt.figure(figsize=(8,5)) #"figure.figsize" : 가로, 세로 인치단위
plt.title("공부시간 대비 점수")
plt.xlabel('점수')
plt.ylabel('시간')
plt.scatter(x, y)  #점그래프
plt.show()

plt.scatter(x, y)
plt.plot([min(x_data), max(x_data)], [min(y_pred), max(y_pred)])
plt.show()