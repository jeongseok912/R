# 상관분석, 회귀분석
# 우리는 종종 어떤 두 사건 간의 연관성을 분석해야 할 경우가 많습니다.
# 둘 또는 그 이상의 변수들이 서로 관련성을 가지고 변화할 때
# 그 관계를 분석하는데 사용되는 방법 중에서 가장 잘 알려진 것이
# 상관분석(correlation analysis)과 회귀분석(regression analysis)입니다.


# 1. 상관분석
# : 상관분석은 변수들이 서로 얼마나 밀접하게 직선적인 관계를 가지고 있는지를 분석하는 통계적 기법
# - 산점도의 점들의 분포를 통해 일정한 패턴을 확인한 후, 상관계수를 구하여 두 변수 간의 선형관계를 확인한다.
# - ex) GDP와 기대수명 간의 관계, 키와 몸무게 간의 관계 등
# - 여기에서 두 사건, 즉 두 변수 간의 선형적 관계를 상관(correlation)이라고 하며
# - 이러한 관계에 대한 분석을 상관분석(correlation analysis)이라고 합니다.


# df데이터
# : 총500명에 대해 8가지 만족도 설문조사
# - 놀이동산에 대한 만족도
# - weekedn(주말이용여부), num.child(동반자녀수),
# - distance(놀이공원까지의 거리), rides(놀이기구에 대한 만족도), games(게임에 대한 만족도),
# - wait(대기시간에 대한 만족도), clean(청결상태에 대한 만족도), overall(전체만족도)로 구성.
df <- read.csv("http://goo.gl/HKnl74")
str(df)
df
# 상관분석을 할 때 결측값(NA)가 있으면 결과가 NA 값이 나오게 되므로 이를 꼭 확인하여 처리.
colSums(is.na(df))
attach(df)

# '놀이기구에 대한 만족도(rides)'와 '전체만족도(overall)' 간의 관계를 분석
# 놀이기구에 대한 만족도가 높으면 전체 만족도 또한 높지 않을까 예상.

# 1) 산점도 그리기
# - 산점도(scatter plot): 직교 좌표계를 이용해 두 개 변수 간의 관계를 나타내는 방법
# - 산점도로부터 두 변수 간에 관련성을 그림을 이용하여 시각적으로 파악.
# - plot(Y~X) 함수의 Y, X에 변수를 입력하면 산점도 그래프 생성.
plot(overall ~ rides)
# ⇒그래프를 통해 봤을 때 대략 양의 관계 확인.

# plot()
# : 산점도 뿐만 아니라 일반적으로 객체를 시각화 하는 데 모두 사용
# - main="Overall~Rides": 그림의 main 제목 입력
# - xlab="Satisfaction with Rides": x축 레이블 입력
# - ylab="Overall Satisfaction": y축 레이블 입력
# - cex=1: 출력되는 점들의 크기 결정
# - pch=1: 출력되는 점의 형태  (기본형태는 빈원)
# - col='red': 색상 지정
# - xlim=: x축 값의 출력범위 지정
# - ylim=: y축 값의 출력범위 지정
# - lty: 출력되는 선의 형태를 지정
plot(overall ~ rides, main="놀이기구에 대한 만족도와 전체만족도", xlab="놀이기구 만족도",
     ylab="전체만족도", cex=1, pch=1, col="red")


# 2) 공분산 및 상관계수
# (1) 공분산(Covariance)
# : 2개의 확률변수의 상관정도를 나타내는 값
# : 두 확률 변수가 얼마나 함계 변하는지 측정
# - 2개의 변수 중 하나의 값이 상승하는 경향을 보일 때 다른 값도 상승하면 공분산 값은 양수(+)
# - 반대로 다른 값이 하강하는 경향을 보이면 공분산의 값은 음수(-)
# - 두 변수 값이 서로 상관없이 움직인다면 공분산은 0
# - 어느정도의 상관관계인지만 확인이 어렵다.
# - 두 변수의 단위에 의존하여 다른 데이터와 비교 시 불편하다.
# - 그래서 공분산을 표준화 시킨 상관계수를 사용한다.
# - 공식 
#     cov(X, Y) = σＸＹ = E[(X - E(Y))(Y - E(Y))]
# - 분산은 Var(X) = E[(X - E(X))²]이므로, 공분산은 분산을 2개 변수로 확장한 형태라고 생각할 수 있다.
# - 함수
#     cov()
cov(1:5, 2:6) # 두 값이 같이 증가하므로 양의 값
cov(1:5, c(3,3,3,3,3)) # 한쪽 값의 변화에 다른 값이 영향을 받지 않아 0
cov(1:5, 5:1) # 값의 증가 방향이 달라 음의 값

cov(overall, rides)
cov(rides, overall) # 이게 맞는 거 아닌가?
# ⇒ 50.82939는 양수이므로, 두 변수 간의 상관관계는 상승하는 경향.


# (2) 상관계수(Correlation Coefficient)
# : 두 변수 간 관련성의 정도
# : 표준화된 공분산
# - 공분산은 각 변량의 단위에 의존하게 되어 변동 크기량이 모호하므로,
#   공분산을 각 변량의 표준편차로 나누어 표준화
# - 종류 : 피어슨/스피어만/켄달 상관 계수
# - 보통 상관 계수라고하면 피어슨 상관 계수를 뜻한다.


# ㄱ) 피어슨 상관계수(Pearson Correlation Codfficient)
# - 두 변수 간의 선형적 상관관계를 측정한다.
# - 계수 범위 : -1 ~ 1
# - -1에 가까울수록 강한 음의 상관관계
# - 0이면 선형 상관관계 아님
# - 1에 가까울수록 강한 양의 상관관계
# - 결과값이 -1 or 1일때는 의미가 없다.
# - -0.7 ~ -0.9, 0.7 ~ 0.9 정도가 더 좋다.
# - 단, 인과관계까지는 유추할수 없고 단지 상관성이 있을 것이라는 추측만 가능
# - Y = aX + b 같은 선형 형태의 관계를 잘 찾는다.
# - Y = aX^2 + b 같은 비선형 관계에서는 피어슨 상관계수가 작게 나타날 수 있다.
# - 공식
#     ρ(X,Y) = cov(X, Y)/σＸσＹ
#   cov(X, Y)는 X, Y의 공분산, σＸ, σＹ는 X, Y의 표준 편차
# - 공분산을 σＸσＹ로 나눠 그 값이 [-1, 1] 사이가 되도록 만들어준 것으로 볼 수 있다.
# - 함수
#     cor(
#       x, # 숫자 벡터, 행렬, 데이터 프레임
#       y=NULL, # NULL, 벡터, 행렬 또는 데이터 프레임
#       method=c("pearson", "kendall", "spearman") # 계산할 상관계수의 종류 지정, 기본값은 피어슨
#     )
#     반환값은 상관계수
cor(iris$Sepal.Width, iris$Sepal.Length)
# ⇒ 큰 상관관계는 없지만, Sepal.Width가 커짐에 따라 Sepal.Length가 작아지는 경향
cor(iris[,1:4])

symnum(cor(iris[,1:4])) # symnum() : 숫자를 심볼로 표현한다.
# ⇒ P.L(Petal.Length)과 Ptetal.Width가 상관계수가 가장 큼(B)
# ⇒ S.L과 Petal.Length, S.L과 Petal.Width가 다음그로 큼(+)

install.packages("corrgram")
library(corrgram)
corrgram(iris, upper.panel = panel.conf) # corrgram() : 상관계수 행렬을 그림으로 보여준다.

cor(overall, rides)
cor(rides, overall) # 이게 맞는거 아닌가?


# ㄴ) 상관계수(상관관계) 검정

# * 통계적 유의성
# : 통계적으로 유의하다는 말은 관찰된 현상이 전적으로 우연에 의해 벌어졌을 가능성이 낮다는 의미
# - 상관계수의 통계적 유의성을 보려면 귀무가설을 '상관계수가 0이다', 대립가설을 '상관계수가 0이 아니다'로 놓은 뒤 p-value를 구한다.
# - p-value가 0.05보다 작다면 귀무가설이 참이라고 가정했을 때, 데이터로부터 구한 상관계수를 볼 확률이 낮다는 의미이며,
#   그런 상관계수는 귀무가설 하에서는 우연히 발생하기 어렵다.
# - 따라서 통계적으로 유의미한 상관계수다.

# cor.test()
# : 상관 계수에 대한 가설 검정을 수행한다.
# - 형식
#     cor.test(
#       x,  # 숫자 벡터
#       y,  # 숫자 벡터
#       alternative=c("two.sided", "less", "greater"),  # 대립가설. 기본값은 양측 검정(two.sided)
#       method=c("pearson", "kendall", "spearman")      # 상관 계수의 종류. 기본값은 피어슨
#     )
cor.test(c(1,2,3,4,5), c(1,0,3,4,5), method = "pearson")
cor.test(c(1,2,3,4,5), c(1,0,3,4,5), method = "kendall")
cor.test(c(1,2,3,4,5), c(1,0,3,4,5), method = "spearman")
# ⇒ 피어슨 상관 계수에서만 p-value가 0.05보다 작아 상관관계가 유의

cor.test(overall, rides)
# ⇒ 귀무가설 "상관관계가 없다"에 대한 검정 결과 p-value < 2.2e-16 값이 나왔으므로 귀무가설을 기각.(?맞나?)
# 그 외에 검정통계량의 값(t), 95% 신뢰구간, 표본상관계수 등을 확인

# 3) 다양한 상관관계 시각화 그래프
cor(df[,4:8])

# (1) R Graphics 패키지 pairs() 함수 이용한 산점도 행렬 그리기
plot(df[,4:8])
pairs(df[,4:8], panel=panel.smooth) # 추세선 그리기

# (2)
install.packages("PerformanceAnalytics")
library(PerformanceAnalytics)

chart.Correlation(df[,4:8], histogram=TRUE, pch=19)

# (3) corrplot 패키지를 이용한 상관계수 행렬 그림 그리기
install.packages("corrplot")
library(corrplot)

df_cor <- cor(df[,4:8])
corrplot(df_cor) # 타원 크기, 색으로 표시
corrplot(df_cor, method="number") # 숫자로 표시
# ⇒ 상관계수가 '0'이면 '선형관계'는 없는 것이지만 '비선형관계'는 있을 수도 있음에 유의.
# ⇒ 두번째 그림에서 보면 상관계수가 '1'인 것은 x, y가 선으로 되어있는 관계이며(기울기는 상관없음), 
#    y 값이 '0'인 선은 상관계수 값이 없음.
# ⇒ 참고) http://rfriend.tistory.com/126


# 4) 다중공선성
# : 회귀분석에서 독립변수들 간에 강한 상관관계가 나타나는 문제
# ?어떤 독립 변수가 다른 독립 변수들과 완벽한 선형 독립이 아닌 경우?
# : 회귀 분석에서 사용된 모형의 일부 설명 변수가 다른 설명 변수와 상관 정도가 높아,
#   데이터 분석 시 부정적인 영향을 미치는 현상.
# - 회귀분석에만 사용. 수치형으로 구성된 데이터에만 사용.

# 설명 변수들끼리 서로 독립이라는 가정.
# 알아보고자 하는 변수의 영향력만을 확인하기 위해.
# 만약 어느 두 설명 변수가 서로에게 영향을 주고 있다면?
# 둘 중 하나의 영향력을 검증할 때 다른 하나의 영향력을 완벽히 통제할 수 없다.


# ex) 회귀 분석을 통해서 음주가 학업 성취도에 미치는 영향
# - 종속 변수 Y : 학업 성취도, 
#   독립 변수 X1 :  일평균 음주량 
#   독립 변수 X2 :  혈중 알코올 농도
# - 평균 음주량이나 혈중 알코올 농도가 높을수록 학업 성취도가 낮아질 것으로 예상
# - 일평균 음주량(X1)과 혈중 알코올 농도(X2)가 완벽하게 서로 독립인가요?
#   두 변수 사이에 일말의 상관관계도 없을까요?
#   일평균 음주량이 높은데 어떻게 혈중 알코올 농도가 높지 않을 수 있을까요?
# - 둘 사이에는 강한 상관관계가 있을 것이다.

# (1) 진단법
#  ① 결정계수 R²값이 높아 설명력은 높지만, 식 안의 독립변수의 p-value 값이 커서
#     개별 인자들이 유의하지 않는 경우 독립변수들 간에 높은 상관관계가 있다고 의심된다.
#  ② 독립변수들 간의 상관계수를 구한다.
#  ③ 분산팽창요인(Variance Inflation Factor)를 구하여 값이 10을 넘으면 보통 다중공선성 문제가 있다.

# (2) 해결법
#  ① 상관관계가 높은 독립변수 중 하나 혹을 일부를 제거한다.
#  ② 변수를 변형시키거나 새로운 관측치를 이용한다.
#  ③ 자료를 수집하는 현장의 상황을 보아 상관관계의 이유를 파악하여 해결한다.
#  ④ PCA(Principle Component Analysis)를 이용한 diagnol matrix 형태로 공선성을 없애준다.

# - 다중공선성을 판단하기 위해 VIF 가 가장 많이 사용된다. 
# - VIF 값이 10 이상 이면 해당 변수가 다중공선성이 존재하는 것으로 판단.
#   1 에서 10 미만이면 다중공선성이 별 문제가 되지 않는 것으로 판단.


# 5) 회귀분석에서의 검정 통계량
# - 귀무가설을 기각하기 위해서는 p-value가 유의 수준보다 작아야 한다.
# - 회귀분석에서 p-value를 구하기 위한 검정 통계량 값은 다음과 같이 구한다.
# - 공식
#     검정 통계량 = (추정된 회귀 계수-0) / 그 계수의 표준 오차
#     T-statistics = (Estimated Beta - H0) / Standard Error
# - 검정 통계량의 절대값이 클수록 p-value는 작아져서 귀무가설을 기각할 수 있게 된다.