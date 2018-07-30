# Ⅰ. 문서 분류
#   - 문서 분류(Document Classification)는 주어진 문서를 하나 이상의 분류로 구분하는 문제다.
#   - 이메일을 보고 해당 이메일이 스팸인지 아닌지를 구분하는 것이 문서 분류의 가장 흔한 예다.
#   - 또 다른 예로는 제품 리뷰 글을 보고 해당 리뷰가 제품에 대한 긍정적인 리뷰인지 부정적인 리뷰인지를 구분하는 
#     감성 분석Sentiment Analysis이 있다.
#   - 텍스트 마이닝(Text Mining) 패키지인 tm을 사용한 문서 분류 방법에 대해 설명한다.
  # 1. tm
  #   - 문서 집합 : Corpus
  #   - 각 문서 : TextDocument
  #   - tm::summary : 코퍼스의 요약 정보를 보여준다.
  # ###############################################################################################
  #   tm::summary(
  #   corpus  # 정보를 살펴볼 corpus
  #   )
  # ###############################################################################################
  #   - tm::inspect : 문서 정보를 보여준다.
  # ###############################################################################################
  #   tm::inspect(
  #   x  # 코퍼스 또는 단어-문서(term-document) 행렬
  #   )
  # ###############################################################################################
  # 2. 예제
  # 로이터(Reuter) 뉴스 기사 중 원유와 관련된 기사 20개가 저장된 crude 데이터를 살펴보자
  install.packages("tm")
  library(tm)
  data(crude)
  summary(crude)
  View(crude)
  # - 문서의 본문은 inspect( ) 함수로 볼 수 있다.
  # - inspect(crude)를 호출하면 모든 문서에 대한 내용을 보여준다.
  # - 특정 문서를 지정해서 보려면 crude[start:end] 형태로 범위를 지정하거나 crude[index] 형태로 색인을 지정한다.
  
  # 다음은 crude의 첫 번째 문서를 살펴보는 예다.
  inspect(crude[1])
  inspect(crude[[1]])
  inspect(crude[2])
  inspect(crude[[2]])
  
  
# Ⅱ. 문서 변환
#   - 어떤 문서에 Computer라는 단어가 있다고 하자.
#     일반적으로 이 단어는 computer, COMPUTER와 같은 단어일 것이다.
#     또, 복수 형태인 computers와도 같은 단어다.
#     뿐만 아니라 ‘computer’와 같이 따옴표가 붙어 있어도 같은 단어다.
#   - 이들을 모두 같은 단어로 처리하려면 문서에서 문장 부호를 제거하거나,
#     문자를 모두 소문자로 바꾸거나, 단어를 그 원형이 되는 뿌리(root) 형태로 바꿔주는
#     스테밍(Stemming)을 적용하는 등 문서를 변환할 필요가 있다.
#     이때 사용하는 함수가 tm_map( )이다.
#   - tm::tm_map : 문서에 함수를 적용하여 변환한다.
#   ###############################################################################################
#     tm::tm_map(
#       x,   # 코퍼스
#       FUN  # 변환에 사용할 함수
#     )
#     반환 값은 변환된 결과다.
#   ###############################################################################################
#   - tm_map에 지정하는 함수는 사용자가 직접 만든 함수여도 되고, tm이 제공하는 함수여도 된다. 
#   - tm이 제공하는 변환 함수들의 목록은 getTransformations( )로 볼 수 있다. 
  # 1. tm의 몇 가지 함수
  getTransformations( )
  # ###############################################################################################
  #   tm::removePunctuation
  #   문장 부호를 없앤다.
  # ###############################################################################################
  #   tm::stripWhitespace
  #   불필요한 공백을 지운다. 연속된 공백 여러 개는 공백 하나로 치환된다.
  # ###############################################################################################
  # 2. 예제
    # 1) crude 문서들의 글자들을 모두 소문자로 바꾸고, 문장 부호를 제거
    crude_rm <- inspect(tm_map(tm_map(crude, tolower), removePunctuation)[1])
    crude_rm[[1]]
    # 2) 위의 내용에서 공백 제거
    crude_str <- inspect(tm_map(tm_map(crude_rm, tolower), stripWhitespace)[1])
    
    
# Ⅲ. 문서의 행렬 표현
#   - 문서를 분류하려면 문서를 기술하는 표현을 문서로부터 추출하고,
#     이로부터 분류를 예측하는 알고리즘을 만들어야 한다.
#   - 문서의 행렬 표현 방식은 이러한 목적으로 가장 많이 사용되는 기법이다.
  # 1. 단어-문서 행렬과 문서-단어 행렬
  #   - 단어와 문서의 행렬로 Corpus를 표현할 때는 TermDocumentMatrix( ) 또는 DocumentTermMatrix( )를 사용한다.
    # (1) TermDocumentMatrix( ) : 주어진 문서들로부터 단어를 행, 문서를 열로 하는 행렬
    # (2) DocumentTermMatrix( ) : 반대로 문서를 행, 단어를 열로 표현한다.
  #   - tm::TermDocumentMatrix : 코퍼스로부터 단어-문서 행렬을 만든다.
  # ###############################################################################################
  #   tm::TermDocumentMatrix(
  #     x,  # 코퍼스
  #     # 제어 옵션
  #     # - bounds: 태그가 global인 리스트로 단어의 최소, 최대 허용 출현 횟수를 지정한다.
  #     #   예를 들어, list(global=c(3, 10))은 3회 미만 또는 10회 이상 발견된 단어를 제외한다.
  #     #   기본값은 모든 단어를 포함시키는 list(global=c(1, Inf))다.
  #     # - weighting: 행렬의 각 셀에 저장할 값을 계산하는 가중치 함수를 지정한다. 기본값은 단어의
  #     #   출현 횟수를 세는 weightTf다.
  #     #   이외에도 weightTfIdf, weightBin, weightSMART를 지정할 수 있다.
  #     control=list(),
  #     ...  # weighting에 추가로 넘겨줄 인자
  #   )
  #   반환 값은 단어-문서 행렬이다.
  # ###############################################################################################
  #   - tm::DocumentTermMatrix : 코퍼스로부터 문서-단어 행렬을 만든다.
  # ###############################################################################################
  #   tm::DocumentTermMatrix(
  #     x,  # 코퍼스
  #     # control과 ...의 의미는 TermDocumentMatrix()와 같다.
  #     control=list(),
  #     ...,
  #   )
  #   반환 값은 문서-단어 행렬이다.
  # ###############################################################################################
  
  # * weighting 함수 중 가장 많이 사용되는 함수는 TF-IDF.
  #   - TF-IDF(TF * IDF) 값이 크면 해당 단어가 문서를 더 잘 기술한다는 의미
    # 1) TF(Term Frequency) :  단어의 출현 횟수
      # (1) TF 계산하는 방법
        # ① 어떤 단어가 그 문서에 출현하는지 여부 TRUE/FALSE로 지정하는 방법
        #   - 출현 빈도 수는 구분하지 않는다. 출현 여부만 봄
        # ② 로그 함수 사용하는 방법
    # 2) IDF(Inverse Document Frequency) : 단어가 출현한 문서 수의 역수

  # 2. 예제
  # 다음은 crude를 단어-문서의 행렬로 표현한 예다.
  (x <- TermDocumentMatrix(crude))
  # A term-document matrix (1266 terms, 20 documents)
  #  
  # Non-/sparse entries: 2255/23065           비/희박 항목.
  # Sparsity           : 91%                  희박
  # Maximal term length: 17                   최대연결.
  # Weighting          : term frequency (tf)  가중치: 용어 빈
  # ⇒ 위 결과를 보면 단어-문서 행렬의 차원이 1266×20이고 행렬의 값은 TF,
  #   즉 단어의 출현 빈도임을 알 수 있다.
    
  # 행렬의 내부는 inspect( )를 사용해서 볼 수 있다.
  inspect(x[1:10, 1:10])
  
  # 다른 가중치 기법 사용하고 싶다면 TermDocumentMatrix() control 인자에 weighting을 지정한다.
  # TF-IDF를 사용한 예
  x <- TermDocumentMatrix(crude, control = list(weighting = weightTfIdf))
  inspect(x[1:10, 1:5])

  
# Ⅳ. 빈번한 단어
#   - tm::findFreqTerms : 단어-문서, 문서-단어 행렬로부터 빈번히 출현하는 단어를 찾는다.
# ###############################################################################################
#   tm::findFreqTerms(
#     x,            # 단어-문서 또는 문서-단어 행렬
#     lowfreq=0,    # 최소 출현 횟수
#     highfreq=Inf, # 최대 출현 횟수. 기본값은 무한대
#   )
#   반환 값은 lowfreq 이상, highfreq 이하 출현하는 빈번한 단어들이다.
# ###############################################################################################

# 다음은 전체 20개 문서로 구성된 crude 코퍼스에서 10회 이상 출현한 단어를 찾은 예다.
findFreqTerms(TermDocumentMatrix(crude), lowfreq=10)
# 행렬에서 전체 단어와 문서의 목록은 rownames( ), colnames( )로 볼 수 있다.
x <- TermDocumentMatrix(crude)
head(rownames(x))
head(colnames(x))


# Ⅴ. 단어 간 상관관계
#   - tm::findAssocs : 주어진 단어와 상관 계수가 높은 단어들을 찾는다.
# ###############################################################################################
#   tm::findAssocs(
#     x,        # 단어-문서 또는 문서-단어 행렬
#     terms,    # 상관 계수가 높은 단어를 찾을 단어들
#     corlimit  # 상관 계수의 하한
#   )
#   반환 값은 단어간 상관 계수가 높은 단어들이다.
# ###############################################################################################

# 다음은 oil과 상관 계수가 0.7 이상인 단어들을 찾은 예다.
# - opec, winter, market, prices 등의 단어를 보면 직관적으로도
#   oil과 함께 출현하는 빈도가 높을 만한 단어들임을 쉽게 예상할 수 있다.
findAssocs(TermDocumentMatrix(crude), "oil", 0.7)


# Ⅵ. 문서 분류
#   - 로이터 기사 중 원유(crude) 토픽(주제)과 인수 합병(acq) 토픽에 대한 문서들을
#     훈련 데이터로 하여, 주어진 문서가 crude와 acq 중 어느 토픽에 속하는지 분류해주는
#     모델을 만들어본다.
  # 1. 데이터 준비
  #   - crude 토픽의 문서와 acq 토픽의 문서는 각각 같은 이름의 코퍼스에 저장되어 있다.
  #   - 따라서 이들을 하나로 합친 뒤 합쳐진 데이터에 대해 분류를 시도할 것이다.
    # 1) 아래 코드는 DocumentTermMatrix( )를 사용하여 crude, acq를 문서-단어 행렬로 만든 뒤 합친다.
    #   - 그 다음 이를 행렬, 데이터 프레임으로 변환해나가면서
    #     LABEL 컬럼에 crude, acq 레이블을 붙이면 모델링에 적합한 형태가 된다.
    data(crude)
    data(acq)
    to_dtm <- function(corpus, label) {
        x <- tm_map(corpus, tolower)
        x <- tm_map(corpus, removePunctuation)
        return(DocumentTermMatrix(x))
      }
    crude_acq <- c(to_dtm(crude), to_dtm(acq))
    crude_acq_df <- cbind(
      as.data.frame(as.matrix(crude_acq)),
      LABEL=c(rep("crude", 20), rep("acq", 50)))
    # ⇒ 위 코드에서 DocumentTermMatrix( )를 만들기 전에 tolower를 적용했으므로
    #   LABEL이라는 컬럼 이름은 문서 내 존재하던 단어들과 중복될 염려가 없다.
    # 2) 적절한 데이터 타입이 잘 부여되었는지를 확인하기 위해 str( )로 crude_acq_df를 살펴보자.
    str(crude_acq_df)
    str(crude_acq_df$LABEL)
    str(crude_acq_df$LABEL[21:30])
    # ⇒ 이 결과를 보면 각 단어에 단어 출현 횟수가 숫자로 저장되어 있고,
    #   각 문서의 분류를 나타내는 LABEL은 팩터로 잘 저장되어 있는 것을 알 수 있다.
  # 2. 모델링
    # 1) 분류 알고리즘을 적용하기에 앞서 훈련 데이터와 테스트 데이터를 나눠보자.
    #   - 80%의 데이터를 훈련 데이터, 20%의 데이터를 테스트 데이터로 나누고,
    #     교차 검증은 사용하지 않았다.
    library(caret)
    train_idx <- createDataPartition(crude_acq_df$LABEL, p=0.8)$Resample1
    crude_acq.train <- crude_acq_df[train_idx, ]
    crude_acq.test <- crude_acq_df[-train_idx, ]
    # 2) rpart를 사용한 의사 결정 나무를 만든다.
    library(rpart)
    m <- rpart(LABEL ~ ., data=crude_acq.train)
    
    confusionMatrix(
      predict(m, newdata=crude_acq.test, type="class"),
      crude_acq.test$LABEL
      )
  # ⇒ 'Positive' Class : acq
  # ⇒ 분석 결과 정확도가 92.86%로 들인 노력에 비해 상당히 괜찮은 모델을 구할 수 있었다.
  #   다만 데이터의 크기가 작아 정확도의 95% CI(Confidence Interval; 신뢰 구간)는
  #   66.13%에서 99.82%로 크게 구해졌다.