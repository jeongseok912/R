# Ⅰ. 군집분석(Clustering Analysis)
#   : 각 객체(대상)의 유사성을 측정하여 유사성이 높은 대상집단을 분류하고, 
#     군집에 속한 객체들의 유사성과 서로 다른 군집에 속한 객체 간의 상이성을 규명하는 분석 방법
#   - 주어진 데이타셋 내에 존재하는 몇 개의 군집을 찾아내는 비지도(unsupervised) 기법이다. 
#   - 군집(cluster)은 다른 그룹에 속한 다른 관찰치들에 비해 서로 보다 유사한 관찰치들의 그룹이라고 할 수 있는데
#     그 정의가 정밀하지 않기 때문에 수 많은 군집 방법이 존재하게 된다.
  # 1. 군집분석의 거리 측정
    # 1) 데이터가 연속형인 경우
    #   - 유클리드 거리, 표준화 거리, 미할라노비스 거리, 체비셔프 거리, 맨하탄 거리, 캔버라 거리, 민코우스키 거리 등 활용
    # 2) 데이터가 범주형인 경우
    #   - 자카드 거리 활용
  # 2. 군집분석 방법
    # 1) Hierarchical agglomerative clustering - 계층적 군집분석
    #   : n개의 군집으로 시작해 점차 군집 개수를 줄여 나가는 방법
    #   - 모든 관찰치는 자신만의 군집에서 시작하여 유사한 데이터 두 개를 하나의 군집으로 묶는데, 
    #     이를 모든 데이터가 하나의 군집으로 묶일때까지 반복한다.
      # 1-1) 사용되는 알고리즘
        # (1) single linkage(최단연결법, nearest neighbor)
        # (2) complete linkage(최장연결법, farthest neighbor)
        # (3) average linkage(평균연결법)
        # (4) controid
        # (5) Ward(와드연결법)
    # 2) Partitioning clustering - 비계층적 군집 분석
    #   : n개의 개체를 g개의 군집으로 나눌 수 있는 모든 가능한 방법을 점검해 최적화한 군집을 형성하는 방법
    #   - 먼저 군집의 갯수 K 를 정한 후 데이타를 무작위로 K개의 군으로 배정한 후 다시 계산하여 군집으로 나눈다.
      # 2-1) 사용되는 알고리즘
      #   - k-means 군집 분석이 가장 많이 사용된다.
        # (1) k-means(K-평균 군집분석)
        # (2) PAM(partitioning around medoids)
  # 3. 알맞은 속성 선택 방법
    # 1) 가장 중요한 단계는 데이타를 군집화하는데 중요하다고 판단되는 속성들을 선택하는 것
    #   - 예를 들어 우울증에 대한 연구라고 하면 다음과 같은 속성들을 평가할 수 있다. 
    #     정신과적 증상, 이학적증상, 발병나이, 우울증의 횟수, 지속기간, 빈도, 입원 횟수, 기능적 상태, 사회력 및 직업력, 
    #     현재 나이, 성별, 인종, 사회경제적 상태, 결혼상태, 가족력, 과거 치료에 대한 반응 등
    #   - 아무리 복잡하고 철저하게 군집분석을 하더라도 잘못 선택한 속성을 극복할 수 없다.
    # 2) 데이타 표준화(Scale the data)(정규화)
    #   - 분석에 사용되는 변수들의 범위에 차이가 있는 경우 가장 큰 범위를 갖는 변수가 결과에 가장 큰 영향을 미치게 된다. 
    #     이런 결과가 바람직하지 않은 경우 데이타를 표준화 할 수 있다. 
    #    - 표준화 : 변수에서 데이터의 평균을 빼거나, 변수를 전체 데이터의 표준 편차로 나누는 작업을 포함한다.
    #               이렇게 하면 변수값의 평균이 0이 되고 값의 퍼짐 정도(분포) 또한 일정해진다.
    #    - 가장 많이 사용되는 방법은 각 변수를 평균 0, 표준편차 1로 표준화하는 것이다.
    #      (x-mean(x))/sd(x)
    #    - scale() 함수 사용
    # 3) 이상치 선별(Screen for outliers)
    #   - 많은 군집분석 방법은 이상치에 민감하기 때문에 이상치가 있는 경우 군집분석 결과가 왜곡된다. 
      # 3-1) 이상치 선별, 제거 방법
        # (1) 단변량 이상치 : outlier 패키지의 함수 사용
        # (2) 다변량 이상치 : mvoutlier 패키지의 함수 사용 
        # (3) 또 다른 방법 : 이상치에 대하여 강건한(robust) 군집분석 방법을 쓸 수 있는데 ,
        #                    PAM(partitioning around medoids)이 대표적인 방법이다.
    # 4) 거리의 계산 (Calcuate distance)
      # 4-1) 두 관찰치 간의 거리를 측정하는 방법
        # (1) euclidean
        # (2) maximum
        # (3) manhattan
        # (4) canberra
        # (5) binary
        # (6) minkowski
    #   - dist()함수 사용(default : 유클리드 거리(eucliean))
    # 5) 군집 알고리즘 선택
    #   - 계층적군집(Hierarchical agglomerative clustering)은 150 관찰치 이하의 적은 데이타에 적합하다. 
    #   - 분할군집은 보다 많은 데이타를 다룰 수 있으나 군집의 갯수를 정해주어야 한다. 
    #   - 계층적 군집/분할 군집을 선택한 후 구체적인 방법을 선택하여야 한다.
    #   - 하나 이상의 군집분석 결과 얻음
    # 6) 군집의 갯수 결정
    #   - 군집분석 최종 결과를 얻기 위해 몇 개의 군집이 있는지 결정한다.
    #   - NbClust 패키지의 NbClust()함수 사용
    # 7) 분석 결과의 시각화
    #   - 계층적 분석은 dendrogram으로 나타내고,
    #     분할군집은 이변량 cluster plot으로 시각화한다.
    # 8) 군집분석 결과의 해석
    #   - 최종 결과를 얻은 후 그 결과를 해석하고 가능하면 이름도 지어야 한다. 
    #     한 군집의 관측치가 갖고 있는 공통점은 무엇인가? 
    #     다른 군집과 어떤 점이 다른가? 
    #     이 단계는 각 군집의 통계량을 요약함으로써 얻어진다. 
    #   - 연속형 변수의 경우 평균 또는 중앙값을 계산,
    #     범주형 변수의 경우 범주별로 각 군집의 분포를 보아야 한다.


# Ⅱ. K-means 군집 분석 알고리즘
  # 1. 방법
    # 1) 분석자가 설정한 K개의 군집 중심점을 랜덤하게 선정
    # 2) 관측치를 가장 가까운 군집 중심에 할당한 후 군집 중심을 새로 계산
    # 3) 기존의 중심과 새로 계산한 군집 중심이 같아질 때까지 반복
  # 2. 데이터 준비
  #     - training 데이터로 모델(70%)을 만들고, 
  #       testing 데이터로 모델(30%)을 평가 하기 위한 사전 작업
  # 3. 유클리드 거리 계산법
  #     dij = √∑pp = 1(xip − xjp)^2
  #     여기서 i,j는 관측치이며, P는 변수 번호이다. 
  # 4. 예제
  # nutrient데이타를 살펴보자.
  install.packages("flexclust")
  library(flexclust)
  data(nutrient, package="flexclust")
  head(nutrient,4)

  # 첫 두 데이터(beef braised와 hamburger)사이의 유클리드 거리
  # d = √(340-245)^2+(20-21)^2+(28-17)^2+(9-9)^2+(2.6-2.7)^2=95.64
  sqrt((340-245)^2+(20-21)^2+(28-17)^2+(9-9)^2+(2.6-2.7)^2)

  # dist()함수는 데이타프레임 또는 행렬의 모든 행 사이의 거리를 계산하여 행렬 형식으로 결과를 반환해 준다. 
  # 다음과 같이 변수 간 거리를 출력한다.
  di <- dist(nutrient)
  as.matrix(di)[1:4,1:4]
  # ⇒ 표준화 전 거리 행렬
  # >              BEEF BRAISED HAMBURGER BEEF ROAST BEEF STEAK
  # > BEEF BRAISED      0.00000   95.6400   80.93429   35.24202
  # > HAMBURGER        95.64000    0.0000  176.49218  130.87784
  # > BEEF ROAST       80.93429  176.4922    0.00000   45.76418
  # > BEEF STEAK       35.24202  130.8778   45.76418    0.00000
  # ⇒ 관측치 사이의 거리가 크다는 것은 관측치가 유사하지 않다는 것이다. 
  # ⇒ 어떤 관측치와 자신과의 거리는 0이다. 
  
  
# Ⅲ. 계층적 군집분석
#   : n개의 군집으로 시작해 점차 군집 개수를 줄여 나가는 방법
#   - 모든 관찰치는 자신만의 군집에서 시작하여 유사한 데이터 두 개를 하나의 군집으로 묶는데, 
#     이를 모든 데이터가 하나의 군집으로 묶일때까지 반복한다.
  # 1. 사용되는 알고리즘
    # 1) single linkage(최단연결법, nearest neighbor)
    # 2) complete linkage(최장연결법, farthest neighbor)
    # 3) average linkage(평균연결법)
    # 4) controid
    # 5) Ward(와드연결법)
  # 2. 방법
    # 1) 모든 관찰치를 군집으로 정의한다.
    # 2) 모든 군집에 대하여 다른 모든 군집과의 거리를 계산한다.
    # 3) 가장 작은 거리를 갖는 두 군집을 합해 하나의 군집으로 만든다. 
    #    따라서 군집의 갯수가 하나 감소한다.
    # 4) 2와3을 반복하여 모든 관찰치가 하나의 군집으로 합쳐질 때까지 반복한다.
  # 3. 군집방법(두 군집 사이의 거리 정의)
  #   - 2단계에서 군집 사이의 거리를 정의하는 것에 따라 계층적 군집 알고리즘이 달라진다. 
  #     가장 많이 쓰이는 다섯가지 방법의 정의는 다음과 같다.
    # 1) single linkage	: 한 군집의 점과 다른 군집의 점 사이의 가장 짧은 거리(shortest distance)
    # 2) complete linkage :	한 군집의 점과 다른 군집의 점 사이의 가장 긴 거리(longest distance)
    # 3) average linkage : 한 군집의 점과 다른 군집의 점 사이의 평균 거리. 
    #                      UPGMA(unweighted pair group mean averaging)이라고도 한다.
    # 4) centroid : 두 군집의 centroids(변수 평균의 벡터) 사이의 거리
    #               관측치가 하나인 경우 centroid는 변수의 값이 된다.
    # 5) Ward	: 모든 변수들에 대하여 두 군집의 ANOVA sum of square를 더한 값
  # 4. 군집방법 특징
    # 1) single linkage clustering 
    #   - 긴 시가모양의 군집이 만들어지는 경향이 있으며, 이러한 현상을 chaining이라고 한다.
    #     chaining은 유사하지 않은 관측치들의 중간 관측치들이 유사하기 때문에 하나의 군집으로 합쳐지는 것을 말한다.
    # 2) complete linkage clustering
    #   - 거의 비슷한 직경을 갖는 compact cluster를 만드는 경향이 있으며 
    #     이상치에 민감한 것으로 알려져 있다. 
    # 3) average linkage clustering
    #   - 두 가지 방법의 타협점이다. 
    #   - chaining 경향이 덜하고 이상치에도 덜 민감하다. 
    #     또한 분산이 적은 군집을 만드는 경향이 있다. 
    # 4) Ward
    #   - 적은 관찰치를 갖는 군집을 만드는 경향이 있으며 
    #     관찰치의 수와 거의 같은 군집을 만드는 경향이 있다. 
    # 5) centroid
    #   - 단순하고 이해하기 쉬운 거리의 정의를 갖는 매력적인 방법으로 
    #    다른 방법들에 비해 이상치에 덜 민감하지만, average linkage나 Ward방법에 비해 수행능력이 떨어진다.
  # 5. R을 이용한 계층적 분석
    # 1) 데이터 준비
    data(nutrient,package="flexclust")
    rownames(nutrient)=tolower(rownames(nutrient))
    # 2) 표준화
    #   - 각 변수를 평균 0, 표준편차 1로 표준화(정규화)
    #     (x-mean(x))/sd(x)
    #   - 표준화 : 변수에서 데이터의 평균을 빼거나, 변수를 전체 데이터의 표준 편차로 나누는 작업을 포함한다.
    #              이렇게 하면 변수값의 평균이 0이 되고 값의 퍼짐 정도(분포) 또한 일정해진다.
    #   - scale() : 행렬 유형의 데이터를 정규화한다.
    #   ###############################################################################################
    #     scale(
    #       x,            # 숫자 벡터 유형의 객체
    #       center=TRUE,  # TRUE면 모든 데이터에서 전체 데이터의 평균을 뺀다.
    #       # scale이 TRUE일 때 center도 TRUE면 모든 데이터를 전체 데이터의 표준 편차로 나눈다.
    #       # scale이 TRUE지만 center는 FALSE면 모든 데이터를 전체 데이터의 제곱 평균 제곱근으로 나눈다.
    #       # scale이 FALSE면 데이터를 어떤 값으로도 나누지 않는다.
    #       scale=TRUE
    #     )
    #   ###############################################################################################
    nutrient.scaled=scale(nutrient)
    nutrient.scaled

    d <- dist(nutrient.scaled)
    as.matrix(d)[1:4,1:4] # 표준화 된 거리행렬
    # 3) 군집화
    #   - hclust() : 계층적 클러스터를 구한다.
    #   ###############################################################################################
    #     hclust(
    #       d, # dist()함수에 의해 만들어지는 거리행렬  
    #       # “ward.D”, “ward.D2”, “single”, “complete”, “average” (= UPGMA), “mcquitty” (= WPGMA), “median” (= WPGMC) 또는 “centroid” (= UPGMC)
    #       method="average"
    #     )
    #   ###############################################################################################
    fit.average <- hclust(d, method="average") # average 방식으로 표준화 된 거리 행렬을 계층적 군집화
    fit.average
    plot(fit.average,hang=-1,cex=.8,main="Average Linkage Clustering")
    # 4) 군집 수 정하기
    install.packages("NbClust")
    library(NbClust)
    
    # 프롬프트가 표시되는지(현재 장치에 대해) 제어하는 데 사용할 수 있습니다.
    devAskNewPage(ask=TRUE)
    
    # 급격히 변하는 선을 보고 군집 개수 예측
    # - NbClust : 여러 가지 군집 갯수 지표(추정 방법)들로 가장 많이 추천된 군집 갯수를 최종으로 추천해준다.
    nc <- NbClust(nutrient.scaled,distance="euclidean",min.nc=2,max.nc=15,
                  method="average") # 유클리드 거리 계산 중 평균거리 계산법 이용하여 군집 수 2~15 범위에 대해  추천 받겠다.
    # ******************************************************************* 
    # * Among all indices:                                                
    # * 4 proposed 2 as the best number of clusters 
    # * 4 proposed 3 as the best number of clusters 
    # * 2 proposed 4 as the best number of clusters 
    # * 4 proposed 5 as the best number of clusters 
    # * 1 proposed 9 as the best number of clusters 
    # * 1 proposed 10 as the best number of clusters 
    # * 2 proposed 13 as the best number of clusters 
    # * 1 proposed 14 as the best number of clusters 
    # * 4 proposed 15 as the best number of clusters 
    
    # ***** Conclusion *****                            
    #   * According to the majority rule, the best number of clusters is  2 
    # *******************************************************************                       
    # ⇒ NbClust의 26개 지표 중 4개의 지표가 2,3,5,15개의 군집 수를 추천하였다.
    # ⇒ According to ~ 는 무시해도 된다.(?)
    # ⇒ 추천받은 4개 군집 중 어떤 걸 선택할지
    #    빨간색 그래프에서는 가장 꺾어지는 부분,
    #    파란색 그래프에서는 가장 튀는 부분을 선택하는 것이 좋다.
    # ⇒ 추천받은 군집 수, 그래프를 봤을 때 군집 수 5개가 가장 유효하다고 보여진다.
    devAskNewPage(ask=FALSE)
    
    table(nc$Best.nc[1,])
    # > 0  1  2  3  4  5  9 10 13 14 15 
    # > 2  1  4  4  2  4  1  1  2  1  4 
    
    par(mfrow=c(1,1))
    barplot(table(nc$Best.n[1,]),xlab="Number of Clusters",ylab="Number of Criteria", main="Number of Clusters Chosen by 26 criteria")
    
    # 5) 찾은 군집 수를 토대로 군집 재분류
    #   - cutree() : 군집 끊기
    clusters<-cutree(fit.average,k=5)
    table(clusters)
    clusters
    # > 1  2  3  4  5 
    # > 7 16  1  2  1 
    
    # 6) 군집 시각화
    aggregate(nutrient,by=list(cluster=clusters),median)
    plot(fit.average,hang=-1,cex=.8,
         main="Average Linkage Clustering\n5 Cluster Solution")
    rect.hclust(fit.average,k=5) # rectangle로 군집 묶어서 시각화
    
  # 6. 계층적 군집화 주의사항
    # 1) 적당한 군집 수를 찾아 나눠주기만 하기 때문에 각 군집의 특성, 군집 생성 이유 등은
    #    개인의 역량으로 알아내야 한다.
    # 2) 그래프와 console 추천 모형이 다를 때가 있다.
    #    이 때는 그래프 우선 순위로 보면 된다.
    # 3) 군집으로 한 번 설정하면 군집 이동이 불가능하다.
    # 4) 거리계산 방법이 달라질 때마다 추천 군집 수가 달라지게 된다.
    #    따라서 어떤 게 제일 나은 지는 군집 내 특성을 분석해봐야 알 수 있다.
    # 5) 관측치가 150개 이하일 때 주로 사용한다. (잘 안 쓰이는 듯??)
    
    
# Ⅳ. 분할 군집분석(비계층적 군집분석)
#   : n개의 개체를 g개의 군집으로 나눌 수 있는 모든 가능한 방법을 점검해 최적화한 군집을 형성하는 방법
#   - 먼저 군집의 갯수 K 를 정한 후 데이타를 무작위로 K개의 군으로 배정한 후 다시 계산하여 군집으로 나눈다.
#   - 사용되는 알고리즘
#     - k-means(K-평균 군집분석)
#     - PAM(partitioning around medoids)
#   - k-means 군집 분석이 가장 많이 사용된다.
  # 1. k-means clustering
    # 1) 알고리즘
      # (1) K개의 centroids(군집 중심점)를 선택한다.
      #     (K개의 행을 무작위로 선택)
      # (2) 각 데이타를 가장 가까운 centroid에 할당한다.
      #     (지정한 점과 남은 점의 모든 거리를 구해 가장 가까운 거리를 구함)
      # (3) 각 군집에 속한 모든 데이타의 평균으로 centroid를 다시 계산한다.
      #     (즉, centroid는 p-개의 길이를 갖는 평균벡터로 p는 변수의 수이다.)
      # (4) 각 데이타를 가장 가까운 centroid에 할당한다.
      # (5) 모든 관측치의 재할당이 일어나지 않거나 최대반복횟수(R에서의 dafault값은 10회)에 도달할 때까지 3과 4를 반복한다.
    # 2) R에서 가장 가까운 centroid 할당 계산법
    #     ss(k) = ∑ni = 1∑pj = 0(xij−x¯kj)^2
    #   - k는 군집이고, xij는 i번째 관측치의 j번째 변수이고, 
    #     x¯kj는 k번째 군집의 j번째 변수의 평균이고, p는 변수의 갯수이다.
    # 3) k-means의 장단점
    #   - 계층적 군집분석에 비해 큰 데이타셋에서 사용할 수 있으며 
    #     관측치가 군집에 영구히 할당되는 것이 아니라 최종결과를 개선시키는 방향으로 이동한다. 
    #   - 평균을 사용하기 때문에 연속형변수에만 적용될 수 있으며 
    #     이상치에 심하게 영향을 받는다. 
    #   - non-convex 형태의(예를 들어, U모양) 군집이 있는 경우 잘 수행되지 않는다.
    # 4) R을 이용한 k-means 군집분석
    #   - k-means 군집 분석을 할 때 무작위로 K개의 행을 선택하므로 실행할 때마다 결과가 달라진다. 
    #   - set.seed()함수를 쓰면 재현 가능한 결과를 얻을 수 있다. 
    #   - K값을 결정하기 위해 계층적 군집분석에서 사용했던 NbClust()함수를 이용할 수 있다.
    install.packages("RGtk2")
    install.packages("rattle")
    library(rattle)
      # (1) 데이터 준비
      wine <- read.csv("data/wine.csv")
      head(wine)
      # (2) 표준화
      df <- scale(wine[-1])
      head(df)
      # (3) 군집 수 정하기
      require(NbClust)
      set.seed(1234)
      nc <- NbClust(df, min.nc=2, max.nc=15, method="kmeans")
      
      table(nc$Best.n[1,])
      # > 0  1  2  3 10 12 14 15 
      # > 2  1  4 15  1  1  1  1 
      par(mfrow=c(1,1))
      barplot(table(nc$Best.n[1,]),xlab="Number of Clusters",ylab="Number of Criteria",
              main="Number of Clusters Chosen by 26 criteria")
      # (4) 군집개수 검증(Elbow point 구하기)
      wssplot <- function(data, nc=15, seed=1234, plot=TRUE){
                  wss <- (nrow(data)-1) * sum(apply(data, 2, var))
                  
                  # 군집 2~15(K)까지 돌면서 withinss(Within cluster sum of squares by cluster)의 합
                  # withins_ss : 같은 군집 내 분산
                  # between_ss : 다른 군집 간의 분산(떨어져 있는 정도)
                  # total_ss : 전체 분산
                  for( i in 2:nc){
                    set.seed(seed)
                    wss[i]<-sum(kmeans(data, centers=i)$withinss)
                  }
                  # ⇒ 잘 분류된 군집일 수록 within_ss 값은 낮고, between_ss는 높다.
                  if(plot) 
                    plot(1:nc, wss, type="b", xlab="Number pf Clusters", ylab="Within group sum of squares")
                  wss
                 }
      wssplot(df)
      # ⇒ 기울기가 급격하게 완만해지는 부분이 보통 elbow point이다.
      #   즉, 그룹의 수에 따른 차이가 별로 나지 않는 갯수를 선택하는 것
      # ⇒ wssplot에서 bend가 있는 곳이 적절한 군집의 갯수를 시사해준다. 
      #   적절한 군집의 갯수가 3개로 판단되므로, 이를 이용해 k-means clustering을 시행한다.
      # (5) 최적의 군집 수로 클러스터링
      fit.km <- kmeans(df, 3, nstart=25)
      fit.km$cluster
      fit.km$size
      fit.km$centers
      aggregate(wine[-1], by=list(clusters=fit.km$cluster),mean)
      # k-means clustering이 Type변수에 저장되어 있는 Type과 어는 정도 일치하는지 평가할 수 있다.
      #=============================================================================
      ct.km <- table(wine$Type, fit.km$cluster) # 이거 어케하지???? 안되는데????
      ct.km
      tail(wine)
      #=============================================================================
  # 2. Partitioning around medoids(PAM)
  #   - k-means clustering 은 평균을 이용하기 떄문에 이상치에 민감한 단점이 있다. 
  #     보다 강건한 방법은 partitioning around medoids(PAM) 방법이다. 
  #   - k-means clustering에서 각 군집을 centroid(변수들의 평균 벡터)로 나타내는 것과 달리 
  #     각 군집은 하나의 관찰치(medoid라고 부른다)로 대표된다. 
  #   - mean은 실제 object가 아니라 그냥 산술적인 평균값이나
  #     medoid는 값이 아리나 가장 평균에 가까운 object이다.
  #   - k-medoids 라고도 한다.
  #   - k-means에서 유클리드 거리를 사용하는 것과 달리 PAM에서는 다른 거리 측정법도 사용할 수 있기 때문에 
  #     연속형 변수들 뿐만 아니라 mixed data type에도 적합시킬 수 있다.
    # 1)  PAM 알고리즘
      # (1) K개의 관찰치(medoid)를 무작위로 선택한다.
      # (2) 모든 관찰치에서 각 medoid까지의 거리를 계산한다.
      # (3) 각 관찰치를 가장 가까운 medoid에 할당한다.
      # (4) 각 관찰치와 해당하는 medoid사이의 거리의 총합(총비용, total cost)을 계산한다.
      # (5) medoid가 아닌 점 하나를 선택하여 그 점에 할당된 medoid와 바꾼다.
      # (6) 모든 관찰치들을 가장 가까운 medoid에 할당한다.
      # (7) 총비용을 다시 계산한다.
      # (8) 다시계산한 총비용이 더 작다면 새 점들을 medoid로 유지한다.
      # (9) medoid가 바뀌지 않을 때까지 5-8단계를 반복한다.
    #   - PAM방법에서 사용하는 수학적인 방법에 대한 예는 https://en.wikipedia.org/wiki/K-medoids를 참조한다.
    
    # 2) R을 이용한 PAM
    #   - cluster 패키지의 pam()함수 이용
    install.packages("cluster")
    library(cluster)
    
    ###############################################################################################
    # pam(
    #   x, # 데이터 행렬 또는 데이터프레임
    #   k, # 군집의 갯수
    #   metric="eucladean", # 거리 측정 방법
    #   stand=FALSE # 거리를 측정하기 전에 변수들을 표준화할 것인지를 나타내는 논리값
    # )
    ###############################################################################################
    
    #=============================================================================
    # wine 데이터에 PAM을 적용해보면 다음과 같다.
    set.seed(1234)
    fit.pam <- pam(wine[-1], k=3, stand=TRUE)
    fit.pam$medoids
    # PAM에서 사용되는 medoid는 wine 데이터에 포함되어 있는 실제 관측치인데 
    # 이 경우 36, 107, 175번쨰 관측치로 이 세 관측치가 세 개의 군집을 대표한다.
    
    clusplot(fit.pam, main="Bivariate Cluster Plot")
    #타원으로 표시되어 있다.
    #이 경우 PAM의 수행능력은 k-means에 비해 떨어진다.
    
    ct.pam=table(wine$Type, fit.pam$clustering) # 이것도 안되네????????????
    
    randIndex(ct.pam)
    ARI 
    0.6994957 
    #=============================================================================
    
  
# 3. iris데이터를 활용하여 군집분석 실행
install.packages("caret")
library(caret)
  
set.seed(1712)
inTrain <- createDataPartition(y = iris$Species, p = 0.7, list = F)
training <- iris[inTrain,]
testing <- iris[-inTrain,]
head(training)
head(testing)
  # 1) 표준화
  #   - K-means 군집 분석은 관측치 간의 거리를 이용하기 때문에 변수의 단위가 결과에 큰 영향을 미친다.
  #   - 그래서 변수를 표준화 하는 작업 필요
  #   - 분류 데이터인 species 데이터를 빼고 실행
  training.data <- scale(training[-5])
  summary(training.data)
  # 2) 모델 작성
  #   - training 데이터를 3개 군집으로 나눈다.(setosa, versicolor, virginica).
  #   - iter.max = 반복의 최대수
  iris.kmeans <- kmeans(training.data[,-5], centers = 3, iter.max = 10000)
  names(iris.kmeans)
  iris.kmeans$centers
  # 3) 군집 확인
  #   - 군집 분석 결과를 training 데이터셋에 할당하고, 결과 확인
  training$cluster <- as.factor(iris.kmeans$cluster)
  qplot(Petal.Width, Petal.Length, colour = cluster, data = training)
  table(training$Species, training$cluster)
  # 4) 군집 갯수 결정
  # - K-means 군집분석에서는 입력하는 변수와 함께 중심의 갯수를 지정하는 것이 중요
  # - 몇 개의 군집 중심이 적당한지 결정하는 방법은 NbClust 패키지를 사용
  install.packages("NbClust")
  library(NbClust)
  
  nc <- NbClust(training.data, min.nc = 2, max.nc = 15, method = "kmeans")
  names(nc)
  nc$Best.partition
  par(mfrow=c(1,1))
  barplot(table(nc$Best.n[1,]),
          xlab="Numer of Clusters", ylab="Number of Criteria",
          main="Number of Clusters Chosen")
  # 5) 검증
  #   - training 데이터셋을 사용해서 예측 모델을 만들고, 
  #     testing 데이터셋으로 모델의 정확성을 다시 한 번 확인
  training.data <- as.data.frame(training.data)
  modFit <- train(x = training.data[,-5],
                  y = training$cluster,
                  method = "rpart")
  
  testing.data <- as.data.frame(scale(testing[-5]))
  testClusterPred <- predict(modFit, testing.data)
  table(testClusterPred ,testing$Species)
  # ⇒ k-means clustering은 평균을 이용하기 떄문에 이상치에 민감한 단점이 있다.