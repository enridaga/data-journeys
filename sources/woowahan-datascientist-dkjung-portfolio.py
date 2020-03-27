
# 필요환경 구성

from __future__ import print_function
import os

data_path = ['../input/intelml101class1']
import numpy as np
import pandas as pd

filepath = os.sep.join(data_path + ['Orange_Telecom_Churn_Data.csv'])
print(filepath)
data = pd.read_csv(filepath)

# 마지막의 예측목표열(churned: 고객이탈)을 포함해 총 21개 지표가 있음 / 사례 고객: 5,000명
data.tail(n=1)
# 기본적 통계수치 확인
data.describe()
# 지표 이름으로 내용 추측
data.columns
# 실제 내용 확인
data.tail(n=1)
# 인덱스로 계정 접근 가능 > account_length 삭제
data.iloc[-1,:]
## 필요없는 열 삭제
col_unused = ['account_length', 'area_code', 'phone_number']
data_slim = data.drop(columns=col_unused)
print(len(data_slim.columns)) # 18 = 21 - 3
data_slim.tail(n=1)
import pandas as pd

# 데이터 불러오기
filepath = os.sep.join(data_path + ['Orange_Telecom_Churn_Data.csv'])
data = pd.read_csv(filepath)
data.head(1).T
# 필요없는 열 삭제
data.drop(['state', 'area_code', 'phone_number'], axis=1, inplace=True)
data.columns
# 현위치 파악
data_slim.tail(n=1).T
# 전체 열 데이터 종류 확인
pd.DataFrame(data_slim).dtypes
# 실험적으로 State부터 변환해보기 / factorize 활용

# 변환 전
print(data_slim.state.tail(n=3))
print()

# 변환 후
data_slim.state = pd.factorize(data_slim.state)[0]
print(data_slim.state.tail(n=3))
data_slim.state.unique()
# 나머지 비숫자 열 변환
for i in range(len(data_slim.columns)):
    if data_slim.dtypes[i] != int and data_slim.dtypes[i] != float: # 숫자 여부 확인
        col = data_slim.columns[i]
        data_slim[col]= pd.factorize(data_slim[col])[0]
data_slim.tail(n=3)
# 변환 결과
data_slim.dtypes
# 추가적으로, 숫자지만 범주에 가까운 열 (number_customer_service_calls) 확인

# 변환 전
print(data_slim.number_customer_service_calls.unique())

# 변환 가능해보임 > 해보자 > 실은 변환할 필요없었음 > 이미 숫자임!
data_slim['number_customer_service_calls'] = pd.factorize(data_slim.number_customer_service_calls)[0]

# 변환 후
print(data_slim.number_customer_service_calls.unique())
# StandardScaler
from sklearn.preprocessing import StandardScaler

"""
잠깐 메모: 파이썬식 변수 이름 짓기 (검색 후 적용)
> 계속 아래_막대_방법을 적용하자,
  작업하고 있는 환경이 엄격하게 camelCase를 따르는 게 아니면
"""
StdSc = StandardScaler()
data_slim_std_scaled = pd.DataFrame(StdSc.fit_transform(data_slim), columns = data.columns)

data_slim_std_scaled.describe().T.tail(n=3)
# 타겟 확인 (스케일링 전) > 0/1 맞음
data_slim.churned.tail(n=5)
# 타겟 확인 (스케일링 후) > 큰일 남!
data_slim_std_scaled.churned.tail(n=5)
# MinMaxScaler (기본 범주인 (0,1) 적용)
from sklearn.preprocessing import MinMaxScaler

#MMSc = MinMaxScaler(feature_range=(0,3)) # 스케일링 범주에 따른 정확도 차이 확인용
MMSc = MinMaxScaler()
data_slim_MMSc_scaled = pd.DataFrame(MMSc.fit_transform(data_slim), columns = data.columns)

data_slim_MMSc_scaled.describe().T.tail(n=3)
# MaxAbsScaler
from sklearn.preprocessing import MaxAbsScaler

MASc = MaxAbsScaler()
data_slim_MASc_scaled = pd.DataFrame(MASc.fit_transform(data_slim), columns = data.columns)

data_slim_MASc_scaled.describe().T.tail(n=3)
# 수치화 후 스케일링 전 데이터 min/max 확인 > 모든 열 >= 0, 모든 min = 0
data_slim.describe().T
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()

for col in ['intl_plan', 'voice_mail_plan', 'churned']:
    data[col] = lb.fit_transform(data[col])
# sklearn 경고 끄기
import warnings
warnings.filterwarnings('ignore', module='sklearn')

from sklearn.preprocessing import MinMaxScaler

msc = MinMaxScaler()

data = pd.DataFrame(msc.fit_transform(data),
                    columns=data.columns)
data.describe().T.tail(n=3)
# LabelBinarizer로 변환해보기

# 실험용 데이터 복사
data_lb = data_slim.copy()

from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()

for column in ['intl_plan', 'voice_mail_plan', 'churned']:
    data_lb[column] = lb.fit_transform(data[column])

data_lb.tail(n=1).T
# 타겟 변환 여부 상세 확인
data_lb.churned
# 현위치 파악
print(data_slim_MMSc_scaled.columns) # 그나저나 이 변수 이름이 너무 길지 않음?;
data_slim_MMSc_scaled['churned'].tail(n=3)
# 위에서 배운 것(drop)부터 적용해보기
y_MMSc = data_slim_MMSc_scaled['churned'] # 이름 짧게
X_MMSc = data_slim_MMSc_scaled.drop(columns='churned')

print(y_MMSc.tail(n=3).T)
X_MMSc.tail(n=3).T.tail(n=3)
from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier(n_neighbors=3)
KNN = KNN.fit(X_MMSc, y_MMSc)

y_predict = KNN.predict(X_MMSc)
print(y_predict)

# 한 뼘 더: KNN에 탑재된 score 기능을 활용해서 정확도 측정
accuracy = KNN.score(X_MMSc, y_MMSc)
print(accuracy)
# 타겟을 제외한 열 목록
x_cols = [x for x in data.columns if x != 'churned']

# 데이터 두 개로 나누기
X_data = data[x_cols]
y_data = data['churned']

# # 다른 방법:
# X_data = data.copy()
# y_data = X_data.pop('churned')
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

knn = knn.fit(X_data, y_data)

y_pred = knn.predict(X_data)
y_pred
# pop으로 데이터 쪼개기
popped_X = data.copy() # 원래 데이터는 살려두기
popped_y = popped_X.pop('churned')

print(popped_y.tail(n=3))

popped_X.tail(n=3).T.tail(n=3)
# 정확도 측정기
def get_accuracy(prediction, target):
    return sum(prediction == target) / float(len(prediction))

get_accuracy(y_predict, y_MMSc)
# 맞는 예측 비율 계산기

def accuracy(real, predict):
    return sum(y_data == y_pred) / float(real.shape[0])
print(accuracy(y_data, y_pred))
# 1. 아마 정확도 측정기에 차이가 있지 않을까
# 두 데이터(인텔, 우리)를 한 측정기(우리)에 적용해보자
print(get_accuracy(y_data, y_pred), get_accuracy(y_MMSc, y_predict))
# 측정기 차이 없음
# 2. 두 예측값 자체가 얼마나 다른가
get_accuracy(y_pred, y_predict)
# 어느 정도 다르긴 하지만, 이 한 수치만으론 원인 파악이 힘듦 (인텔/우리 각각의 정확도와도 다름)
# 3. 고객이탈(1)로 예측된 값을 모아보자 (총 5000개 사례 중 얼마나 되는지)
print(sum(y_pred), sum(y_predict)) # 예측값
print(sum(y_data), sum(y_MMSc)) # 실제값
# 이탈 고객 수 총합만으론 개별 고객 예측값과 실제값의 일치 여부를 알 수 없음
# 4. 각 예측값이 각 실제값과 얼마나 같은가
print(sum(y_pred==y_data), sum(y_predict==y_MMSc))
# 위의 수치로 볼 때 인텔보다 우리 모델이 11명 더 많은 고객에 대해 정확한 예측을 내놓는 것을 확인
# 하지만 여전히 이 차이의 '원인'을 모르겠음
# 우리는 트레이닝 데이터에서 'account_length'를 삭제
# 이 열이 가질 수 있는 값의 종류는 218 가지 / 각 값이 갖는 평균 고객의 수는 약 23명
print(len(data.account_length.unique()), len(data)/len(data.account_length.unique()))
# 반면 인텔은 'state'를 필요없다고 보고 삭제
# 이 열이 가질 수 있는 값의 종류는 51 가지 / 각 값이 갖는 평균 고객의 수는 약 98명
print(len(data_slim.state.unique()), len(data_slim)/len(data_slim.state.unique()))
# 잠시 쉬는 줄 (이건 읽지 마시고 눈을 쉬게 해주세요!)
# 위 내용과 관련하여 질문이 있다면 누구든지 주저말고 아래 댓글로 달아주세요
## 그동안 kaggle 내에서 숨어 살았는데 이번 기회(?)에 처음으로 이렇게 공개 활동을 해보네요
### 워낙 고수분들이 많은 프로들의 세계라 겁도 나지만 설레기도 합니다
#### 긴 글 읽느라 고생 많으셨습니다. 따뜻한 연말 보내세요