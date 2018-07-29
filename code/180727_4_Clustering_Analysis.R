# <군집분석>
# 군집분석은 주어진 데이타셋 내에 존재하는 몇 개의 군집을 찾아내는 비지도(unsupervised) 기법이다. 군집(cluster)는 다른 그룹에 속한 다른 관찰치들에 비해 서로 보다 유사한 관찰치들의 그룹이라고 할 수 있는데 그 정의가 정밀하지 않기 때문에 수 많은 군집 방법이 존재하게 된다.
# http://rfriend.tistory.com/228

# 군집분석 방법
# 1) Hierarchical agglomerative clustering
# 모든 관찰치는 자신만의 군집에서 시작하여 유사한 데이타 두 개를 하나의 군집으로 묶는데 이를 모든 # 데이타가 하나의 군집으로 묶일때까지 반복한다.
# 사용되는 알고리즘 : single linkage, complete linkage, average linkage, controid, Ward의 방법 등이 있다.

# 2) Partitioning clustering
# 먼저 군집의 갯수 K 를 정한 후 데이타를 무작위로 K개의 군으로 배정한 후 다시 계산하여 군집으로 나눈다.
# 사용되는 알고리즘 : k-means 및 PAM(partitioning around medoids) 등이 있다.


# 알맞은 속성 선택(Choose appropriate attributes)
# 1) 가장 중요한 단계는 데이타를 군집화하는데 중요하다고 판단되는 속성들을 선택하는 것
# 예를 들어 우울증에 대한 연구라고 하면 다음과 같은 속성들을 평가할 수 있다. 
# 정신과적 증상, 이학적증상, 발병나이, 우울증의 횟수, 지속기간, 빈도, 입원 횟수, 기능적 상태, 사회력 및 직업력, 현재 나이, 성별, 인종, 사회경제적 상태, 결혼상태, 가족력, 과거 치료에 대한 반응 등. 
# 아무리 복잡하고 철저하게 군집분석을 하더라도 잘못 선택한 속성을 극복할 수 없다.

# 2) 데이타 표준화(Scale the data)
# 분석에 사용되는 변수들의 범위에 차이가 있는 경우 가장 큰 범위를 갖는 변수가 결과에 가장 큰 영향을 미치게 된다. 
# 이런 결과가 바람직하지 않은 경우 데이타를 표준화 할 수 있다. 
# 가장 많이 사용되는 방법은 각 변수를 평균 0, 표준편차 1로 표준화하는 것이다.
# (x-mean(x))/sd(x)
# R의 scale() 함수를 사용

# 3) 이상치 선별(Screen for outliers)
# 많은 군집분석 방법은 이상치에 민감하기 때문에 이상치가 있는 경우 군집분석 결과가 왜곡된다. 
# 단변량 이상치의 경우 outlier 패키지의 함수를 사용할 수 있고 
# 다변량 이상치의 경우 mvoutlier 패키지에 있는 함수들을 사용하여 이상치를 선별하고 제거할 수 있다. 
# 또 다른 방법은 이상치에 대하여 강건한(robust) 군집분석 방법을 쓸 수 있는데 PAM(partitioning around medoids)이 대표적인 방법이다.

# 4) 거리의 계산 (Calcuate distance)
# 두 관찰치 간의 거리를 측정하는 방법은 여러가지가 있는데 
# “euclidean”, “maximum”, “manhattan”, “canberra”, “binary” 또는 “minkowski” 방법을 사용할 수가 있다. 
# R의 dist()함수를 쓰면 위의 방법들을 이용하여 거리를 계산할 수 있으며 
# 디폴트 값은 유클리드 거리(“euclidean”)이다. 

# 5) 군집 알고리즘 선택
# 다음 단계로 군집 방법을 선택하여야 한다. 
# 계층적군집(Hierarchical agglomerative clustering)은 150 관찰치 이하의 적은 데이타에 적합하다. 

# 분할군집은 보다 많은 데이타를 다룰 수 있으나 군집의 갯수를 정해주어야 한다. 
# 계층적 군집/분할 군집을 선택한 후 구체적인 방법을 선택하여야 한다.
# 하나 이상의 군집분석 결과 얻음

# 6) 군집의 갯수 결정
# 군집분석 최종 결과를 얻기 위해 몇 개의 군집이 있는지 결정해야 한다. 
# NbClust패키지의 NbClust()힘수를 사용할 수 있다. 

# 8) 분석 결과의 시각화
# 최종 결과를 시각화할 때 계층적 분석은 dendrogram으로 나타내고 분할군집은 이변량 cluster plot으로 시각화한다.

# 9) 군집분석 결과의 해석
# 최종 결과를 얻은 후 그 결과를 해석하고 가능하면 이름도 지어야 한다. 한 군집의 관측치가 갖고 있는 공통점은 무엇인가? 다른 군집과 어떤 점이 다른가? 이 단계는 각 군집의 통계량을 요약함으로써 얻어진다. 
# 연속형 변수의 경우 평균 또는 중앙값을 계산하고 범주형 변수가 있는 경우 범주별로 각 군집의 분포를 보아야 한다.


# 1. 군집 분석은 유사한 대상끼리 그룹핑 하는 분석.
# - 주어진 데이타셋 내에 존재하는 몇 개의 군집을 찾아내는 비지도(unsupervised)기법.
# 2. K-means 군집 분석의 알고리즘
#  1) 분석자가 설정한 K개의 군집 중심점을 랜덤하게 선정
#  2) 관측치를 가장 가까운 군집 중심에 할당한 후 군집 중심을 새로 계산
#  3) 기존의 중심과 새로 계산한 군집 중심이 같아질 때까지 반복
# 3. 데이터 준비
#  - training 데이터로 모델(70%)을 만들고, testing 데이터로 모델(30%)을 평가 하기 위한 사전 작업.
# 유클리드 거리는 다음과 같은 공식으로 계산된다.

# dij=√∑pp=1(xip−xjp)2

# 여기서 i,j는 관측치이며 P는 변수 번호이다. 
# nutrient데이타를 살펴보자.
install.packages("flexclust")
library(flexclust)
data(nutrient,package="flexclust")
head(nutrient,4)
# 첫 두 데이타(beef braised와 hamburger)사이의 유클리드 거리는 다음과 같이 구할 수 있다.
# d=√(340−245)2+(20−21)2+(28−17)2+(9−9)2+(2.6−2.7)2=95.64

#R의 dist()함수는 데이타프레임 또는 행렬의 모든 행 사이의 거리를 계산하여 행렬 형식으로 결과를 반환헤 준다. 다음과 같이 할 수 있다.
# >              BEEF BRAISED HAMBURGER BEEF ROAST BEEF STEAK
# > BEEF BRAISED      0.00000   95.6400   80.93429   35.24202
# > HAMBURGER        95.64000    0.0000  176.49218  130.87784
# > BEEF ROAST       80.93429  176.4922    0.00000   45.76418
# > BEEF STEAK       35.24202  130.8778   45.76418    0.00000

#관측치 사이의 거리가 크다는 것은 관측치가 유사하지 않다는 것이다. 어떤 관측치와 자신과의 거리는 0이다. beef braised와 hamburger 사이의 거리는 손으로 계산한 값과 같다.


#<계층적군집 방법>
#모든 관찰치는 자신만의 군집에서 시작하여 유사한 데이타 두 개를 하나의 군집으로 묶는데 이를 모든 데이타가 하나의 군집으로 묶일때까지 반복한다. 알고리즘은 다음과 같다.

# 1. 모든 관찰치를 군집으로 정의한다.
# 2. 모든 군집에 대하여 다른 모든 군집과의 거리를 계산한다.
# 3. 가장 작은 거리를 갖는 두 군집을 합해 하나의 군집으로 만든다. 따라서 군집의 갯수가 하나 감소한다.
# 4. 2와3을 반복하여 모든 관찰치가 하나의  군집으로 합쳐질 때까지 반복한다.
#2단계에서 군집 사이의 거리를 정의하는 것에 따라 계층적 군집 알고리즘이 달라진다. 가장 많이 쓰이는 다섯가지 방법의 정의는 다음과 같다.

#군집방법	두 군집 사이의 거리 정의
# 1) single linkage	한 군집의 점과 다른 군집의 점 사이의 가장 짧은 거리(shortest distance)
# 2) complete linkage	한 군집의 점과 다른 군집의 점 사이의 가장 긴 거리(longest distance)
# 3) average linkage	한 군집의 점과 다른 군집의 점 사이의 평균 거리. UPGMA(unweighted pair group mean averaging)이라고도 한다.
# 4) centroid	두 군집의 centroids(변수 평균의 벡터) 사이의 거리.관측치가 하나인 경우 centroid는 변수의 값이 된다
# 5) Ward	모든 변수들에 대하여 두 군집의 ANOVA sum of square를 더한 값

# 1) single linkage clustering은 긴 시가모양의 군집이 만들어지는 경향이 있으며 이러한 현상을 chaining이라고 한다. chaining은 유사하지 않은 관측치들의 중간 관측치들이 유사하기 때문에 하나의 군집으로 합쳐지는 것를 말한다.

# 2) complete linkage clustering은 거의 비슷한 직경을 갖는 compact cluster를 만드는 경향이 있으며 이상치에 민감한 것으로 알려져 있다. 

# 3) average linkage clustering은 두 가지 방법의 타협점이다. chaining결향이 덜하고 이상치에도 덜 민감하다. 또한 분산이 적은 군집을 만드는 경향이 있다. 
# 4) Ward의 방법은 적은 관찰치를 갖는 군집을 만드는 경향이 있으며 관찰치의 수와 거의 같은 군집을 만드는 경향이 있다. 
# 5) centroid방법은 단순하고 이해하기 쉬운 거리의 정의를 갖는 매력적인 방법으로 다른 방법들에 비해 이상치에 덜 민감하지만 average linkage나 Ward방법에 비해 수행능력이 떨어진다.

# R을 이용해 계층적 분석을 하려면 다음과 같이 한다.
# hclust(d, method=)
# d는 dist()함수에 의해 만들어지는 거리행렬이고 method로는 “ward.D”, “ward.D2”, “single”, “complete”, “average” (= UPGMA), “mcquitty” (= WPGMA), “median” (= WPGMC) 또는 “centroid” (= UPGMC)를 사용할 수 있다.

data(nutrient,package="flexclust")
rownames(nutrient) <- tolower(rownames(nutrient))
nutrient.scaled=scale(nutrient)

d <- dist(nutrient.scaled)
fit.average <- hclust(d, method="average")
plot(fit.average,hang=-1,cex=.8,main="Average Linkage Clustering")

#몇 개의 군집으로 나누어야 하는가?
install.packages("NbClust")
library(NbClust)

# 프롬프트가 표시되는지 (현재 장치에 대해) 제어하는 데 사용할 수 있습니다.
devAskNewPage(ask=TRUE)

# 급격히 변하는 선을 보고 군집 개수 예측
nc <- NbClust(nutrient.scaled,distance="euclidean",min.nc=2,max.nc=15,
              method="average")
devAskNewPage(ask=FALSE)

table(nc$Best.n[1,])

# > 0  1  2  3  4  5  9 10 13 14 15 
# > 2  1  4  4  2  4  1  1  2  1  4 
par(mfrow=c(1,1))
barplot(table(nc$Best.n[1,]),xlab="Number of Clusters",ylab="Number of Criteria", main="Number of Clusters Chosen by 26 criteria")

clusters<-cutree(fit.average,k=5)
table(clusters)
clusters
# > 1  2  3  4  5 
# > 7 16  1  2  1 

aggregate(nutrient,by=list(cluster=clusters),median)
plot(fit.average,hang=-1,cex=.8,
     main="Average Linkage Clustering\n5 Cluster Solution")
rect.hclust(fit.average,k=5)

#####################################################################
#####################################################################
#분할군집
#분할군집에서는 먼저 군집의 갯수 K 를 정한 후 데이타를 무작위로 K개의 군으로 배정한 후 다시 계산하여 군집으로 나눈다. k-means clustering과 PAM을 다룬다.

#k-means clustering
#k-means 알고리즘
#분할 군집에서 가장 많이 사용되는 방법은 k-means clustering이다. 알고리즘은 다음과 같다.

# 1. K개의 centroids를 선택한다.(K개의 행을 무작위로 선택)
# 2. 각 데이타를 가장 가까운 centroid에 할당한다.
# 3. 각 군집에 속한 모든 데이타의 평균으로 centroid를 다시 계산한다.(즉, centroid는 p-개의 길이를 갖는 평균벡터로 p는 변수의 수이다.)
# 4. 각 데이타를 가장 가까운 centroid에 할당한다.
# 5. 모든 관측치의 재할당이 일어나지 않거나 최대반복횟수(R에서의 dafault값은 10회)에 도달할 때까지 3과 4를 반복한다.
# R에서 가장 가까운 centroid에 할당할때 다음 값을 이용하여 계산한다.

# ss(k)=∑ni=1∑pj=0(xij−x¯kj)2
# 여기서 k는 군집이고 xij는 i번째 관측치의 j번째 변수이고 x¯kj는 k번째 군집의 j번째 변수의 평균이고 p는 변수의 갯수이다.

# k-means의 장단점
#k-means는 계층적 군집분석에 비해 큰 데이타셋에서 사용할 수 있으며 관측치가 군집에 영구히 할당되는 것이 아니라 최종결과를 개선시키는 방향으로 이동한다. 
# 하지만 평균을 사용하기 때문에 연속형변수에만 적용될 수 있으며 이상치에 심하게 영향을 받는다. 
#또한 non-convex형태의(예를 들어 U모양) 군집이 있는 경우 잘 수행되지 않는다.

# R을 이용한 k-means 군집분석
# k-means 군집 분석을 할 때 무작위로 K개의 행을 선택하므로 실행할 때마다 결과가 달라진다. set.seed()함수를 쓰면 재현 가능한 결과를 얻을 수 있다. K값을 결정하기 위해 계층적 군집분석에서 사용했던 NbClust()함수를 이용할 수 있다.
install.packages("RGtk2")
install.packages("rattle")
library(rattle)
data(wine) 
#wine <- read.csv("./data/winequality-red.csv")
wine <- read.csv("./data/wine.csv")
head(wine)

df <- scale(wine[-1])
head(df)
require(NbClust)
set.seed(1234)
nc <- NbClust(df,min.nc=2,max.nc=15,method="kmeans")

table(nc$Best.n[1,])

# > 0  1  2  3 10 12 14 15 
# > 2  1  4 15  1  1  1  1 
par(mfrow=c(1,1))
barplot(table(nc$Best.n[1,]),xlab="Number of Clusters",ylab="Number of Criteria",
        main="Number of Clusters Chosen by 26 criteria")

wssplot <- function(data,nc=15,seed=1234,plot=TRUE){
  wss <- (nrow(data)-1)*sum(apply(data,2,var))
  
  for( i in 2:nc){
    set.seed(seed)
    wss[i]<-sum(kmeans(data,centers=i)$withinss)
  }
  if(plot) plot(1:nc,wss,type="b",xlab="Number pf Clusters",ylab="Within group sum of squares")
  wss
}

wssplot(df)

#wssplot에서 bend가 있는 곳이 적절한 군집의 갯수를 시사해준다. 적절한 군집의 갯수가 3개로 판단되므로 이를 이용해 k-means clustering을 시행한다.

fit.km <- kmeans(df,3,nstart=25)
fit.km$cluster

fit.km$size

fit.km$centers

aggregate(wine[-1],by=list(clusters=fit.km$cluster),mean)

# k-means clustering이 Type변수에 저장되어 있는 Type과 어는 정도 일치하는지 평가할 수 있다.

ct.km <- table(wine$type, fit.km$cluster)
ct.km
################################################################
#################################################################
# Partitioning around medoids(PAM)
#k-means clustering 은 평균을 이용하기 떄문에 이상치에 민감한 단점이 있다. 보다 강건한 방법은 partitioning around medoids(PAM) 방법이다. k-means clustering에서 각 군집을 centroid(변수들의 평균 벡터)로 나타내는 것과 달리 각 군집은 하나의 관찰치(medoid라고 부른다)로 대표된다. k-mean에서 유클리드 거리를 사용하는 것과 달리 PAM에서는 다른 거리 측정법도 사용할 수 있기 때문에 연속형 변수들 뿐만 아니라 mixed data type에도 적합시킬 수 있다.

# PAM 알고리즘
# K개의 관찰치(medoid)를 무작위로 선택한다.
# 모든 관찰치에서 각medoid까지의 거리를 계산한다.
# 각 관찰치를 가장 가까운 medoid에 할당한다.
# 각 관찰치와 해당하는 medoid사이의 거리의 총합(총비용,total cost)을 계산한다.
# medoid가 아닌 점 하나를 선택하여 그 점에 할당된 medoid와 바꾼다.
# 모든 관찰치들을 가장 가까운 medoid에 할당한다.
# 총비용을 다시 계산한다.
# 다시계산한 총비용이 더 작다면 새 점들을 medoid로 유지한다.
# medoid가 바뀌지 않을 때까지 5-8단계를 반복한다.
# PAM방법에서 사용하는 수학적인 방법에 대한 예는 https://en.wikipedia.org/wiki/K-medoids 를 참조한다.
xtabs(~ wine$type + fit.km$cluster)

#R을 이용한 PAM
#cluster패키지의 pam()함수를 이용하여 PAM 을 시행할 수 있다. 다음과 같은 형식으로 사용한다.

pam(x,k,metric="eucladean", stand=FALSE)
#여기서 x는 데이터 행렬 또는 데이터프레임이고 k는 군집의 갯수, metric은 거리 측정 방법이고 stand는 거리를 측정하기 전에 변수들을 표준화할 것인지를 나타내는 논리값이다. wine 데이터에 PAM을 적용해보면 다음과 같다.

library(cluster)
set.seed(1234)
fit.pam <- pam(wine[-1],k=3,stand=TRUE)
fit.pam$medoids
#PAM에서 사용되는 medoid는 wine 데이터에 포함되어 있는 실제 관측치인데 이 경우 36, 107, 175번쨰 관측치로 이 세 관측치가 세개의 군집을 대표한다.

clusplot(fit.pam, main="Bivariate Cluster Plot")

#타원으로 표시되어 있다.

#이 경우 PAM의 수행능력은 k-means에 비해 떨어진다.

ct.pam=table(wine$Type, fit.pam$clustering)

randIndex(ct.pam)
ARI 
0.6994957 
################################################################
#################################################################

# 4. iris데이터를 활용하여 군집분석 실행
library(caret)
set.seed(1712)

inTrain <- createDataPartition(y = iris$Species, p = 0.7, list = F)
training <- iris[inTrain,]
testing <- iris[-inTrain,]
head(training)

# 5. 표준화
# K-means 군집 분석은 관측치 간의 거리를 이용하기 때문에 변수의 단위가 결과에 큰 영향을 미친다.
# - 그래서 변수를 표준화 하는 작업  필요. scale 함수를 사용해서 표준화.

# 분류 데이터인 species 데이터를 빼고 실행.
training.data <- scale(training[-5])
summary(training.data)
# 6. 모델 작성
# training 데이터를 3개 군집으로 나눈다(setosa, versicolor, virginica).
# iter.max = 반복의 최대수
iris.kmeans <- kmeans(training.data[,-5], centers = 3, iter.max = 10000)

names(iris.kmeans)
iris.kmeans$centers

# 7. 군집 확인
# 군집 분석 결과를 training 데이터셋에 할당하고, 결과를 확인.
training$cluster <- as.factor(iris.kmeans$cluster)
qplot(Petal.Width, Petal.Length, colour = cluster, data = training)
table(training$Species, training$cluster)

# K-means 군집분석에서는 입력하는 변수와 함께 중심의 갯수를 지정하는 것이 중요
# - 몇개의 군집 중심이 적당한지 결정하는 방법은 NbClust 패키지를 사용.
#install.packages("NbClust")
library(NbClust)

nc <- NbClust(training.data, min.nc = 2, max.nc = 15, method = "kmeans")
names(nc)
nc$Best.partition
par(mfrow=c(1,1))
barplot(table(nc$Best.n[1,]),
        xlab="Numer of Clusters", ylab="Number of Criteria",
        main="Number of Clusters Chosen")

# training 데이터 셋을 사용해서 예측 모델을 만들고, testing 데이터 셋으로 모델의 정확성을 다시 한번 확인해 보겠습니다.

training.data <- as.data.frame(training.data)
modFit <- train(x = training.data[,-5],
                y = training$cluster,
                method = "rpart")

testing.data <- as.data.frame(scale(testing[-5]))
testClusterPred <- predict(modFit, testing.data)
table(testClusterPred ,testing$Species)

# k-means clustering은 평균을 이용하기 떄문에 이상치에 민감한 단점이 있다.

# Partitioning around medoids(PAM)
# 각 군집은 하나의 관찰치(medoid라고 부른다)
# 연속형 변수들 뿐만 아니라 mixed data type에도 적합

# cluster패키지의 pam()함수
#
# 모형식; pam(x,k,metric="eucladean", stand=FALSE)
# x는 데이터 행렬 또는 데이터프레임
# k는 군집의 갯수
# metric은 거리 측정 방법
# stand는 거리를 측정하기 전에 변수들을 표준화할 것인지를 나타내는 논리값.
library(cluster)
set.seed(1234)
fit.pam <- pam(wine[-1],k=3,stand=TRUE)
fit.pam$medoids
