library(dplyr)
library(tidyverse)
library(stringr)
library(tidyr)
library(dummies)
library(tibble)
library(fastDummies)
library(ggfortify)
library(readr)
library(ggplot2)
library(gridExtra)
library(cluster)
library(scales)
library(rio)
library(psych)
library(plyr)
library(caTools)
library(rpart)
library("caret")
library("randomForest")
library("e1071")
library("MLmetrics")
library(rpart.plot)
library(class)
library("MLeval")
library(rattle)
library(xlsx)
library(factoextra)
library('funModeling')
library(useful)
library(kohonen)
library(bnlearn)
library("pROC")
library(Rcpp) 
library(forecast)
library(Rtsne)
BiocManager::install("Rgraphviz")

#load data
campus <- read.csv("Campus_placement.csv")

#missing value computation
sapply(campus,function(x) sum(is.na(x)))
missing <- campus %>%
  gather(key = "col_name", value = "value") %>%
  mutate(is_missing = is.na(value)) %>%
  group_by(col_name, is_missing) %>%
  summarise(missing_count = n()) %>%
  mutate(percentage = missing_count / 215 * 100)%>%
  filter(is_missing==T) %>%
  select(-is_missing)

#Data analysis of categorical variable
insight<- freq(data=campus, input = c('gender','ssc_b','hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation', 'status'), plot = FALSE )

glimpse(campus)
psych::describe(campus)

#dropping irrelevant columns that are: hsc_b, ssc_b and salary
data<- campus%>% select(-ssc_b,-hsc_b,-salary)
#Dummy variables for gender, ssc_b, hsc_b,hsc_s, degree_t, workex, specialisation 
data <- dummy_cols(data, select_columns = c("gender","hsc_s","degree_t","workex","specialisation"))
data<- data %>% select(-gender,-hsc_s,-degree_t,-specialisation)
#factor status
data$status<- factor(data$status)

#Classification
#Random forest
#train <- data[1:107, ]
#test <- data[108:215, ]
train_data <- createDataPartition(data$status, p = 0.8, list = FALSE, times = 1)

train <- data[train_data,]
test <- data[-train_data,]

train_x <- train %>% dplyr::select(-status)
train_y <- train$status

test_x <- test %>% dplyr::select(-status)
test_y <- test$status

training <- data.frame(train_x, target = train_y)

random_f <- randomForest(train_x, train_y, importance = TRUE)
random_f

important<-varImp(random_f)
write.xlsx(important,"C:/Users/SALONI/Documents/Semester 3/Data/Project/important.xlsx")
varImpPlot(random_f)

pred<-predict(random_f, test_x)
head(pred)
confusionMatrix(pred, test_y)


#Decision tree

tree <- rpart(status ~ ., data =train)

pred_1 <- predict(tree, test, type = "class")

confusionMatrix(as.factor(test$status),as.factor(pred_1))

fancyRpartPlot(tree)

featureimp <- data.frame(tree$variable.importance)
featureimp$features <- rownames(featureimp)
featureimp <- featureimp[, c(2, 1)]
featureimp$importance <- round(featureimp$tree.variable.importance, 2)
featureimp$rpart.tree.variable.importance <- NULL
featureimp %>%
  ggplot(aes(x=reorder(features, importance), y=importance, fill=features)) + 
  geom_bar(stat='identity') +
  ggtitle("Feature Importance") + 
  labs(x = "Variable") + labs(y = "Value")+theme(legend.position = "none") 

#SVM

svm_model<- svm(status~.,train, type = 'C-classification', kernel = 'linear')

pred_2<-predict(svm_model,test)

confusionMatrix(as.factor(test$status),as.factor(pred_2))

summary(svm_model)
#Naive bayes
bayes_model <- naiveBayes(status~., data = train, laplace = 1)
bayes_model$apriori
pred_3 <- predict(bayes_model, test)
confusionMatrix(pred_3, test$status)

#Clustering
#K means 
v_keep <- c("hsc_p","ssc_p","etest_p","mba_p")
M <- data%>% select(hsc_p,ssc_p,etest_p,mba_p)
scale_M <- scale(M)

kmeans_campus_2 <- kmeans(  
  x = scale_M,  
  centers = 2
)
kmeans_campus_2

kmeans_campus_3<- kmeans(  
  x = scale_M,  
  centers = 3 
)
kmeans_campus_3

kmeans_campus_4<- kmeans(  
  x = scale_M,  
  centers = 4 
)
kmeans_campus_4

kmeans_campus_5<- kmeans(  
  x = scale_M,  
  centers = 5 
)
kmeans_campus_5

set.seed(823)
factoextra::fviz_nbclust(  x = scale_M,  FUNcluster = kmeans,  method = "wss")

set.seed(823)
factoextra::fviz_nbclust(  x = scale_M,  FUNcluster = kmeans,  method = "silhouette")

#scatter plot
useful::plot.kmeans(  x = kmeans_campus_2,  data = scale_M  )

kmeans_clustering<- kmeans(scale_M, 2)
kmeans_clustering_2 <- fviz_cluster(list(data=scale_M, cluster=kmeans_campus_2$cluster),
                                    ellipse.type="convex", 
                                    geom="point",stand=FALSE) + labs(title = "Plot for K-means Clustering")+
                                    theme_minimal() +scale_color_brewer(palette="Dark2") 
kmeans_clustering_2

kmeans_clustering_1<- kmeans(scale_M, 3)
kmeans_clustering_3 <- fviz_cluster(list(data=scale_M, cluster=kmeans_campus_3$cluster),
                                    ellipse.type="convex", 
                                    geom="point",stand=FALSE) + labs(title = "Plot for K-means Clustering")+
                                    theme_minimal() +scale_color_brewer(palette="Dark2") 
kmeans_clustering_3


#Partitioning methods
fviz_nbclust(scale_M,FUNcluster=clara,method = "s")
clara_M <- cluster::clara(  x = scale_M,  k = 2 )
fviz_cluster(clara_M)
plot(clara_M)

fviz_nbclust(scale_M,FUNcluster=pam,method = "s")
pam_M <- cluster::pam(M,k = 2)
fviz_cluster(pam_M)
plot(pam_M)

#Kohonen clustering
set.seed(823)

#rectanguar topology models
som_campus <- som(  X = scale_M[,v_keep],  grid = somgrid(3,3, "rectangular"),  rlen = 1000)
plot(  x = som_campus,  type = "changes",  palette.name = viridis::viridis)
plot(  x = som_campus,  type = "codes",  palette.name = viridis::viridis)

v_legend <- viridis::viridis(  n = length(unique(data$status)))
names(v_legend) <- sort(unique(data$status))
v_color <- v_legend[data$status]
plot(  x = som_campus,  type = "mapping",  col = v_color,  pch = 19)
legend(  "topright",  legend = names(v_legend),  col = v_legend,  pch = 19)

plot(  x = som_campus,  type = "counts",  palette.name = viridis::viridis)
plot(  x = som_campus,  type = "dist.neighbours",  palette.name = viridis::viridis)
plot(  x = som_campus,  type = "quality",  palette.name = viridis::viridis)
j <- "mba_p" 
for(j in c("hsc_p","ssc_p","etest_p","mba_p"))
  plot(  x = som_nba,  type = "property",  property = getCodes(som_nba)[,j],  
         main = paste0("Heat map of ",j),  palette.name = viridis::viridis)

#hexagonal topology models
set.seed(73)
som_campus_hexagonal <- som(  X = scale_M[,v_keep],  grid = somgrid(3, 3, "hexagonal"),  rlen = 1000)
plot(  x = som_campus_hexagonal,  type = "changes",  palette.name = viridis::viridis)

plot(  x = som_campus_hexagonal,  type = "codes",  palette.name = viridis::viridis)
plot(  x = som_campus_hexagonal,  type = "mapping",  col = v_color,pch = 19)
legend(  x = "topright",  legend = names(v_legend),  col = v_legend,  pch = 19)
plot(  x = som_campus_hexagonal,  type = "counts",  palette.name = viridis::viridis)
plot(  x = som_campus_hexagonal,  type = "dist.neighbours",  palette.name = viridis::viridis)
plot(  x = som_campus_hexagonal,  type = "quality",  palette.name = viridis::viridis)
for(j in c("hsc_p","ssc_p","etest_p","mba_p")) 
  plot(  x = som_campus_hexagonal,  type = "property",  property = getCodes(som_campus)[,j],  
         main = paste0("Heat map of ",j),  palette.name = viridis::viridis)

#pca

v_keep= c("hsc_p","ssc_p","etest_p","degree_p", "mba_p")

M = data[, v_keep]
scale_data = scale(M)
scale_data

prcomp_M <- prcomp(scale_data)
prcomp_M
summary(prcomp_M)
str(prcomp_M)
head(prcomp_M$x)

plot(prcomp_M)

screeplot(prcomp_M)

biplot(prcomp_M)

autoplot(
  object = prcomp_M,
  data = data,
  colour = "status"
)

factoextra::fviz_eig(prcomp_M)

factoextra::fviz_pca_ind(
  prcomp_M,
  col.ind = "cos2",
  gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
  repel = TRUE
)

factoextra::fviz_pca_var(
  X = prcomp_M,
  col.var = "contrib",
  gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
  repel = TRUE
)

factoextra::fviz_pca_biplot(
  X = prcomp_M,
  repel = TRUE,
  col.var = "#2E9FDF",
  col.ind = "#696969"
)

factoextra::get_eigenvalue(prcomp_M)
get_pca_var_M <- factoextra::get_pca_var(prcomp_M)
get_pca_var_M
get_pca_var_M$coord
get_pca_var_M$cor
get_pca_var_M$cos2
get_pca_var_M$contrib

get_pca_ind_M <- factoextra::get_pca_ind(prcomp_M)
get_pca_ind_M
head(get_pca_ind_M$coord)
head(get_pca_ind_M$cos2)
head(get_pca_ind_M$contrib)

#t-SNE

y <- data$status

set.seed(823)
Rtsne_1 <- Rtsne::Rtsne(
  X = scale_data,
  dims = 3,
  PCA = FALSE,
  max_iter = 2000,
  perplexity = 3
)

plot(
  Rtsne_1$Y,
  col=y,
  pch = as.character(y),
  main = "Scatterplot of EPL T-SNE two dimensions"
)

prcomp_new <- prcomp(
  x = scale_data,
  center = TRUE,
  scale. = TRUE,
  rank = 2
)

plot(
  prcomp_new$x[,1:2],
  col=y,
  pch = as.character(y),
  main = "Scatterplot of EPL PCA two dimensions",
)
#Hierarchical Clustering

#Distance matrix

dist_M <- dist(scale_M[1:10,])
heatmap( x = as.matrix(dist_M), col = viridis::viridis(256))
factoextra::fviz_dist(dist_M)

#hclust dendogram
hclust_M <- hclust(dist_M)
plot(hclust_M)

#Choosing distance matrix
v_dist <- c("canberra","manhattan","euclidean")
list_dist <- lapply(
  X = v_dist,
  FUN = function(distance_method) dist(
    x = scale_M,
    method = distance_method
  )
)
names(list_dist) <- v_dist
v_hclust <- c("ward.D","single","complete")

#Selecting Clustering method
list_hclust <- list()
for(j in v_dist) for(k in v_hclust) list_hclust[[j]][[k]] <- hclust(
  d = list_dist[[j]],
  method = k
)

par(
  mfrow = c(length(v_dist),length(v_hclust)),
  mar = c(0,0,0,0),
  mai = c(0,0,0,0),
  oma = c(0,0,0,0)
)

for(j in v_dist) for(k in v_hclust) plot(
  x = list_hclust[[j]][[k]],
  labels = FALSE,
  axes = FALSE,
  main = paste("\n",j,"\n",k)
)


plot(
  x = list_hclust[["manhattan"]][["ward.D"]],
  main = "manhattan Ward's D",
  sub = ""
)

plot(
  x = list_hclust[["euclidean"]][["ward.D"]],
  main = "euclidean ward.D",
  sub = ""
)

d <- dist(scale_M, method = "euclidean")
hc5 <- hclust(d, method = "ward.D")
cutree_M <- cutree(hc5, k = 2)
factoextra::fviz_cluster(list(data = scale_M, cluster = cutree_M))

v <- prcomp(scale_M)$x[,1]
scale_M <- scale_M[order(v),]



