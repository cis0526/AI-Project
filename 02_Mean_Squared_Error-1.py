#-*- coding: utf-8 -*-

import numpy as np

a = []  # 빈 리스트 생성
for k in range(-10,10):
   line = []  # 안쪽 리스트로 사용할 빈 리스트 생성
   line.append(k)  # 안쪽 리스트에 0 추가
   line.append(76)
   a.append(line)  # 전체 리스트에 안쪽 리스트를 추가
print(a)

for i in a :

   #가상의 기울기 a와 y 절편 b
   fake_a_b=i

   # x 값과 y값
   data = [[2, 81], [4, 93], [6, 91], [8, 97]]
   x = [i[0] for i in data]
   y = [i[1] for i in data]

#   print(x)
#   print(y)

   # y=ax + b에 a,b 값 대입하여 결과를 출력하는 함수
   def predict(x):
      return fake_a_b[0]*x + fake_a_b[1]

   # MSE 함수
   def mse(y, y_hat):
      return ((y - y_hat) ** 2).mean()

   # MSE 함수를 각 y값에 대입하여 최종 값을 구하는 함수
   def mse_val(y, predict_result):
      return mse(np.array(y), np.array(predict_result))

   # 예측값이 들어갈 빈 리스트
   predict_result = []

   # 모든 x값을 한 번씩 대입하여 predict_result 리스트완성.
   for t in range(len(x)):
      predict_result.append(predict(x[t]))
 #     print("공부시간=%.f, 실제점수=%.f, 예측점수=%.f" % (x[i], y[i], predict(x[i])))

   #predict_result=([predict(x[i]) for i in range(len(x))])
   # 최종 MSE 출력
   print("기울기가 "+str(i[0])+"일때 MSE 최종값: " + str(mse_val(predict_result,y)))
