install.packages("PASWR")
library(PASWR)
install.packages("rms")
library(rms)
install.packages("ggplot2")
library(ggplot2)
install.packages("gbm")
library(gbm)
install.packages("randomForest")
library(randomForest)
install.packages("caret")
library(caret)
install.packages("vip")
library(vip)
install.packages("default")
library(default)
install.packages("dplyr")
library(dplyr)
install.packages("mlbench")
library(mlbench)

install.packages("superml")
library(superml)
library(tidyverse)
library(dplyr)
library(Metrics) # handy evaluation functions


########
# Part A--------------------------------Data Preprocessing & analysis (with viulizations)
########
set.seed(123)   

#Reading File in to df
data1 <- read.csv("C:/Users/user/Desktop/INFO 4050/movie_metadata.csv",header=TRUE,sep=',')

#looking at the data
#View(data1)

#checking the null counts and there dattypes
na_count <-sapply(data1, function(y) sum(length(which(is.na(y)))))
datatype <-sapply(data1, function(y) typeof(y))
na_count <- data.frame(datatype,na_count)
na_count

#######
#Finding:
#       From the output of above code it can be seen that the dataset has some Nan values in both catagorical 
# and numarical features Removing them in the next step.
#######

#looking at the dimention of the data frame before removing the Nan Vlaues
dim(data1)

#removing the Nan rows from the dataSet
data1 <- na.omit(data1)

#looking at the dimention of the data frame after removing the Nan Vlaues
dim(data1)



#1. content_rating

p1<- ggplot(data1, aes(x=content_rating))+geom_bar(fill="lightblue")+ labs(x="Content Rating")+ theme_minimal(base_size=10)
p1
#############################################################################################
#finding p1:
#it can be seen from the graph that the data set Conatins movies with content rating R highest, then
#PG13 and on the thir rank PG movies.
#############################################################################################

#2 Relatioship Between content_rating and budget 
d2 <-table(data1$content_rating,data1$budget)
p2 <-barplot(d2,main='Relatioship Between content_rating and budget', xlab='content_rating', ylab='budget')
p2
#################################################################################################
#finding p2:
#The Relatioship Between content_rating and budget can be represented by the normal curve as 
#can be seen from the graph 
#################################################################################################


# Droping unimportant columns
drop <- c("color","director_name","actor_2_name","actor_1_name","movie_title","actor_3_name","movie_imdb_link")
data1 = data1[,!(names(data1) %in% drop)]


#checking to see the data types of Remaining column
str(data1)

# Lable encoding the Remaining catagorical features beacuse we want to train machine learning models
label <- LabelEncoder$new()

#Label Encoding genres 
data1$genres <- label$fit_transform(data1$genres)
data1$plot_keywords <-  label$fit_transform(data1$plot_keywords)
data1$language <- label$fit_transform(data1$language)
data1$country <- label$fit_transform(data1$country)
data1$content_rating <- label$fit_transform(data1$content_rating)

#looking at the datatypes again to see if any catagorical varaible remains
na_count <-sapply(data1, function(y) sum(length(which(is.na(y)))))
datatype <-sapply(data1, function(y) typeof(y))
na_count <- data.frame(datatype,na_count)
na_count


# we'd have 90% training data and 10% testing data, which would provide the highest level of accuracy.
# we choosed the above partion as the dataset is ver small 
split1<- sample(c(rep(0, 0.9 * nrow(data1)), rep(1, 0.1 * nrow(data1))))

trainset <- data1[split1 == 0, ]   
testset  <- data1[split1 == 1, ]
dim(trainset)
dim(testset)

########
# Part B------------------------------------ MODELING
########
#################################
#applying machine learning models---------(dependent variable) => imdb_score
#################################

####################
#1.Linear Regression
####################

#Training on trainset
library(ggplot2)
lm1 <- lm(imdb_score ~ num_critic_for_reviews + duration + director_facebook_likes + gross + genres + num_voted_users + cast_total_facebook_likes + facenumber_in_poster + plot_keywords + num_user_for_reviews + language + country + content_rating + budget + title_year + aspect_ratio + movie_facebook_likes, data=trainset)
summary(lm1)
#ploting the regression model
ggplot(lm1)

#plotting the histogram of dependent varaibele
ggplot(data=data1, aes(imdb_score)) + 
  geom_histogram(breaks=seq(1, 10, by=0.5), 
                 col="red", 
                 aes(fill=..count..)) +
  scale_fill_gradient("Count", low="green", high="red")+
  labs(title="Histogram imdb_score",x="imdb_score",y="Count")

###############
#2.RandomForest
###############

# Training the model
rforest<- randomForest(factor(imdb_score) ~. , data=trainset, ntree=500, importance=TRUE)

#summary(rforest)
#Looking at the Important Features wrt to Influence
imp<-varImp(rforest)
varImpPlot(rforest)

# Making predictions and also testing on testset
rpredict<- predict(rforest, testset, type="class")

####################
#3.Kmeans Clustring
####################

bet <- numeric()
withn <- numeric()

# Run the algorithm for different values of k 
set.seed(1234)

for(i in 1:20){
  
  # For each k, calculate betweenss and tot.withinss
  bet[i] <- kmeans(trainset, centers=i)$betweenss
  withn[i] <- kmeans(trainset, centers=i)$tot.withinss
  
}

# Between-cluster sum of squares vs Choice of k
p3 <- qplot(1:20, bet, geom=c("point", "line"), 
            xlab="Number of clusters", ylab="Between-cluster sum of squares") +
  scale_x_continuous(breaks=seq(0, 20, 1)) +
  theme_bw()

# Total within-cluster sum of squares vs Choice of k
p4 <- qplot(1:20, withn, geom=c("point", "line"),
            xlab="Number of clusters", ylab="Total within-cluster sum of squares") +
  scale_x_continuous(breaks=seq(0, 20, 1)) +
  theme_bw()

# Subplot
grid.arrange(p3, p4, ncol=2)

########
#Finding:
#       The perfect value for k from the above two graphs can be seen as 14 i.e. setting k=14 centers
########

set.seed(1234)
Imbd_k14 <- kmeans(trainset, centers=14)

# Cluster size
Imbd_k14$size

# Total sum of squares
Imbd_k14$totss

rmarkdown::render("C:/Users/user/Desktop/INFO 3010/Final Project 11801090.R")

