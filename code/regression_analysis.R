# 2. 회귀분석
# 한 개 또는 그 이상의 변수들(x, 원인변수, 독립변수)에 대하여 다른 한 변수(종속변수) 사이의 관계를
# 수학적인 모형을 이용하여 설명하고 예측하는 분석기법
# 상관분석의 이 일정한 패턴을 활용하여 무엇인가를 예측하는 분석.

# y=종속변수=결과변수=반응변수=대응변수=왼쪽
# x=독립변수=원인변수=설명변수=요인변수=오른쪽.

# 단순선형회귀분석(simple linear regression analysis):
# -독립변수 종속변수가 각각 한 개일 때의 관계를 분석.

# 다중선형회귀분석(multiple linear regression analysis)
# - 종속변수는 한개 독립변수는 두개 이상일 때.

# 2-1 단순선형회귀분석
# 2-1-1 회귀식의 추정
# 두 변수 X와 Y의 관계(rides, overall)에 적합한 회귀식을 구하기 위해서는
# - 관측된 값으로부터 회귀계수 B0와 B1의 값을 추정.
# 이 때 일반적으로 많이 사용되는 방법을 최소제곱법.

# '놀이기구에 대한 만족도(rides)'와 '전체만족도(overall)' 간의 관계를 분석
lm(overall ~ rides)
# b0 = -94.962, b1 = 1.703 으로부터
# overall = -94.962 + 1.703*rides 라는 회귀식을 구할 수 있으며,
# 놀이기구에 대한 만족도(rides)가 1 증가할 때마다 전체만족도(overall)이 1.703만큼 증가.

# 이렇게 구해진 회귀직선을 산점도로 확인.
m1 <- lm(overall~rides) # m1에 회귀식을 입력
plot(overall~rides) # 산점도를 그림
abline(m1, col='blue') # 산점도 위에 m1이라는 회귀직선을 blue 색으로 그림
summary(m1)

# 2-2-2. 회귀모형의 검정 및 적합도 파악
# 회귀식이 통계적으로 유의한지, 변수가 유의하게 영향을 미치는 지,
# 그리고 얼만큼의 설명력을 가지는지 등의 여부를 확인.

# F-statistic
# 도출된 회귀식이 회귀분석 모델 전체에 대해 통계적으로 의미가 있는지 파악

# P-Value
# 각 변수가 종속변수에 미치는 영향이 유의한지 파악

# 수정된 R제곱

# 회귀직선에 의하여 설명되는 변동이 총변동 중에서 차지하고 있는 상대적인 비율이
# - 얼마인지 나타냄
# 즉, 회귀직선이 종속변수의 몇%를 설명할 수 있는지 확인.
# 결과를 보면 잔차에 대한 정보, 회귀계수에 대한 정보, R제곱, 검정통계량 F0 값과 P-Value 값 등이 출력된 것을 확인할 수 있습니다.

# 제일 밑에 F-statistic의 p-value 값은 2.2e-16 로 0.05보다 작기에
# 이 회귀식은 회귀분석 모델 전체에 대해 통계적으로 의미가 있다.

# 중간의 Coefficients:에는 y절편 값(Intercept) 및 변수들의 p-value 값이 나와있습니다.
# rides 변수의 경우 2e-16으로 0.05보다 작기에 overall을 설명하는데 유의하다고 판단.

# *는 통계적으로 유의하다는 것을 한 눈에 보여주는 표시이며 *가 많을수록
# - 통계적으로 유의할 확률이 높다고 볼 수 있습니다.

# 밑 부분의 Adjusted R-squared 값은 0.3421 로써 34%만큼의 설명력을 가진다고 판단.
# - (0에 가까울 수록 예측값 믿을 수 없고 1에 가까울 수록 믿을 수 있다)

# 2-2 다중선형회귀분석
# overall을 설명하는 독립변수를 기존에 있었던 rides에 + games와 clean 변수를 추가.
# lm(Y~X1+X2+ ... Xn)과 같은 형식으로 입력해주면 회귀식이 나옴.
m2 <- lm(overall ~ rides + games + clean)
summary(m2)

# 회귀식: overall = -131.67877 + 0.57798*rides + 0.26028*games + 1.28381*clean
# 1. 제일 밑에 F-statistic의 p-value 값이 2.2e-16 로 0.05보다 작기에
# - 이 회귀식은 회귀분석 모델 전체에 대해 통계적으로 의미가 있다고 볼 수 있습니다.

# 2. 중간의 Coefficients에 나온 변수들의 p-value 값이 모두 0.05보다 작기에
# - overall을 설명하는데 유의하다고 판단할 수 있습니다.

# 3. 밑 부분의 Adjusted R-squared 값은 0.4358 로써 43.5%만큼의 설명력을 가진다고 판단.
# - 앞선 결과와 비교했을 때 더 높은 설명력을 가진다.


###########################################################################################
###########################################################################################

# 회귀분석; 잔차를 최소화 하는 선형회귀식을 찾는것.

# 1. 단순선형회귀분석(Simple Regression)
# 광고비(cost,독립변수)의 변화가 매출액(sales, 종속변수)에
# 미치는 영향을 살펴보기 위한 모델이다
# 데이터(매출액, 광고비, 60개)
sim_regre <- read.csv("simple regression.csv", header = T)
summary(sim_regre)
head(sim_regre)

# lm : 선형 회귀를 수행한다.
# lm(
#  formula,  # 종속 변수 ~ 독립 변수 형태로 지정한 포뮬러
#  data      # 포뮬러를 적용할 데이터. 보통 데이터 프레임
# )

# 산점도를 통해 선형인지, 비선형인지 확인.
attach(sim_regre)
plot(sales~cost)
abline(sim_regre_lm, col="red")
# abline(sim_regre, col="blue")
sim_regre_lm <- lm(sales~cost, sim_regre)
sim_regre_lm
summary(sim_regre_lm)
# 매출액(sales) = 0.6411 + 0.8134*광고비(cost)
# 절편이 -0.6411, 광고비에 대한 기울기가 0.8134
# sim_regre 데이터의 각 광고비값에 대해 모델에 의해 예측된 매출액값은
# - fitted( )로 구할 수 있다. 이 값은 모델이 데이터에 적합fit된 결과이므로
# - 적합된 값fitted value라고 부른다.
# 다음은 sim_regre 데이터의 1~4번째 데이터에 대한 적합된 값들을 보여준다.

# 잔차(residuals): 선형 회귀 모델을
# 작성한 다음 모델로부터의 구한
# - 예측값과 실제 값 사이의 차이는
# - 잔차residual라고 부른다.

# - 정규분포에서 얼마나 벗어났는지를 알 수있다.
fitted(sim_regre_lm)[1:4]
names(sim_regre_lm)

# 원데이터의 매출액
head(sim_regre$sales)

# 선형회귀분석에 의한 매출액
head(sim_regre_lm$fitted.values)

# 잔차: 원데이터의 매출액과 선형회귀분석에 의한 매출액의 차이.
sim_regre_lm$residuals
head(sim_regre$sales)- head(sim_regre_lm$fitted.values)
# 잔차(sim_regre_lm$residuals) = sim_regre$sales - sim_regre_lm$fitted.values

summary(sim_regre_lm)
# call: 포물라 식
# Residuals: 잔차의 최소값, 1분위수값(25%), 중앙값, 3분위수값(75%), 최대값
# Coefficients:
# 회귀계수의 추정량(Estimate), 표준오차(Std. Error),
# 검정을 위한 t-통계량(t value)과 p-값(Pr(>|t|))을 출력.
#  p-vlaue가 계산된 곳에서 나온 t통계량

# Residual standard error: 잔차의 표준오차(잔차의 표준편차)

# Multiple R-squared(결정계수)
# vs Adjusted R-squared(수정된 결정계수)
# Multiple R-squared(결정계수): 상관계수의 제곱.
# *** 독립변수가 종속변수를 얼마나 잘 설명하고
# 있는가를 나타낸 계수

# 1. 회귀분석으로 만든 모형이 실제 데이터에 얼마만큼
# - 잘 적용되는지 나타내는 측도
# 2. 종속변수의 변화(변동)을 얼마나 설명하는지
# - 나타내는 지표
# 3. 결정계수는 0과 1사이의 값.
# 4. 결정계수는 주어진 독립변수들로 예측한
# 종속변수값(예측치)와 실제 종속변수값(관측치)의
# -상관관계를 나타내는 계수
# 5. 적합된 회귀방정식의 설명력:
# 독립변수가 추가되면 항상 증가하게 됨

# Adjusted R-squared(수정된 결정계수):
# *** 다중회귀분석에서는 결정계수가 아닌
# 수정된 결정계수 사용.

# 1. 결정계수 보다는 항상 작다.
# 2. 독립변수가 추가된다고 해서 항상 증가하지 않는다.
# 3. 독립변수의 수(p)가 적고
# 표본의 수(n)이 클수록
# 수정결정계수는 결정계수의 값에 가까워진다.
# 4. 표본에서 얻어진 결정계수의 값은
# 모집단에서 얻어진 결정계수 보다 약간 커진다.
# - 이를 보완해 주는 것이 수정된 결정계수
# 5. 변수의 수가 늘어난다면 결정계수의 값보다
# - 수정된 R-제곱의 값으로 적합도를
# - 판단하는 것이 효율적이다.
# 6. 모형의 적합도를 방해하는 요인이 늘어나게 된다면
# - 오히려 값이 줄어들 수도 있다.

# Pr(>|t|): 각 독립변수의 유의성을 판단하기 위한 통계량
# F-statistic: 통계량은 모형 전체의 유의성을
# - 판단하기 위한 통계량.

# 모형 진단 그래프
par(mfrow = c(2, 2))

plot(sim_regre_lm)

# 첫 번째 그래프는 Residuals vs Fitted plot.
# 잔차의 평균은 0을 중심으로 일정하게 패턴 없이 분포.

# 두 번째 그래프는 Normal Q-Q plot(분위수(Quantile)의 약어) 잔차가 정규분포를 따르는지 확인
# 직선관계가 뚜렷하게 성립해야 한다.

# 세 번째 그래프는 Scale-Location plot.
# 표준화 잔차는 ‘잔차 / 잔차의 표준 편차’로 계산한다.
# - 말 그대로 잔차의 분산을 없애 표준화한 잔차
# 이상점(outlier)을 탐지할 수 있는 그래프로 빨간색 추세선이 0인 직선이 가장 이상적이며
# - 크게 벗어난 값은 이상점일 가능성이 있습니다.

# 네 번째 그래프는 Residuals vs Leverage plot 입니다.
# 여기서 레버리지는 설명변수가 얼마나 극단에 치우쳐 있는지를 말합니다.
# Cook’s distance: 회귀 직선의 모양(기울기나 절편 등)에 크게 영향을 끼치는 점들을 찾는 그래프


###########################################################################################
###########################################################################################

# 2. 다중회귀분석
# 하나 이상의 독립 변수가 사용된 선형 회귀.
# 스마트폰의 사용자가 만족을 느끼는 요인이
# - 외관, 유용성, 편의성 중 어떤 것인지
# - 알아보기 위해 회귀분석을 진행해보자.
# Y = 종속변수 = 만족감(satisfaction)
# X = 독립변수 = 외관(design), 유용성(convenience),
# - 편의성(Usefulness)

multi_regre <- read.csv("multi regression.csv", header = T)
summary(multi_regre)
head(multi_regre)
# rm(multi_regre)
attach(multi_regre)
# detach(multi_regre)

multi_regre_lm <- lm(satisfaction ~ design + convenience + usefulness, multi_regre)
multi_regre_lm

# 만족감(satisfaction) = 1.4583 + 0.1444*cost + 0.2839*convenience + 0.1737*usefulness)
# multi_regre_lm 데이터의 각 항목에 대해 모델에 의해 예측된 만족감은
# - fitted( )로 구할 수 있다. 이 값은 모델이 데이터에 적합fit된 결과이므로
# - 적합된 값fitted value라고 부른다.
# 다음은 multi_regre_lm 데이터의 1~4번째 데이터에 대한 적합된 값들을 보여준다.

# 잔차(residuals): 선형 회귀 모델을 작성한 다음 모델로부터의 구한
# - 예측값과 실제 값 사이의 차이는 잔차residual라고 부른다.
fitted(multi_regre_lm)[1:4]
head(multi_regre$satisfaction)
names(multi_regre_lm)

# 원데이터의 매출액
head(multi_regre$satisfaction)

# 선형회귀분석에 의한 매출액
head(multi_regre_lm$fitted.values)

# 잔차: 원데이터의 매출액과 선형회귀분석에 의한 매출액의 차이.
head(multi_regre_lm$residuals)

# 잔차(multi_regre_lm$residuals) = multi_regre$satisfaction - multi_regre_lm$fitted.values

summary(multi_regre_lm)

# 다중공선성: 회귀 모델의 독립변수(설명변수=x)들
# 사이에 상관관계가 있는 경우가 있는데,
# - 이것을 “다중공선성” 라고 한다.
# (고객 만족도를 조사하면서 상품의 내구도와
# - 불량률을 설명 변수로 동시에 포함시키는 경우0

# 다중공선성이 존재할 경우
# - 모델의 정확도가 하락하게 되므로
# - 어떤 두 변수 간에 다중 공선성이 존재할 경우
# - 설명력이 더 적은 변수를 제거하고
# - 모델을 재구성 한다.

# 다중공선성을 판단하는 방법에는 여러가지가 있지만,
# R에서는 vif 함수를 사용해 VIF값을 간단히 구할 수 있으며,
# 보통 VIF 값이 4~10가 넘으면 다중공선성이 존재
library(car)
vif(multi_regre_lm)
