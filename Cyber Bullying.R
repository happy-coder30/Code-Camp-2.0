# Load required libraries
library(tm)
library(e1071) #Naive Bayes library
library(caret) #Used for Classification And Regression Training
library(wordcloud)
library(dplyr)
library(gmodels)
# Library for parallel processing
library(doMC)
registerDoMC(cores=detectCores())  # Use all available cores


# Read file
mydata <- read.csv(file.choose(), header = T, stringsAsFactors = F)
glimpse(mydata)

#Randomize the dataset
set.seed(1)
mydata <- mydata[sample(nrow(mydata)), ]
mydata <- mydata[sample(nrow(mydata)), ]
glimpse(mydata)

#data in type is currently not represented as categorical data, we can set that manually
mydata$label <- factor(mydata$label)
str(mydata$label) 

#Cleaning and standardizing text data
comments_corpus <- VCorpus(VectorSource(mydata$Comments))
typeof(comments_corpus) # Just to show that it is a list
print(comments_corpus)


comments_corpus_clean <- comments_corpus %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords()) %>%
  tm_map(removePunctuation) %>%
  tm_map(stemDocument) %>%
  tm_map(stripWhitespace)

cat("The text document prior processing:", "\n")
for(i in 1:3){
  print(as.character(comments_corpus[[i]]))
}

cat("\n")
cat("The text document after processing:", "\n")

for(i in 1:3){
  print(as.character(comments_corpus_clean[[i]]))
}

#Splitting text documents into words(Tokenization)
comments_dtm <- DocumentTermMatrix(comments_corpus_clean)

comments_dtm_no_prep <- DocumentTermMatrix(
  comments_corpus,
  control = list(
    tolower = TRUE,
    removeNumbers = TRUE,
    stopwords = TRUE,
    removePunctuation = TRUE,
    stemming = TRUE
  )
)

cat("Our matrix with preprocessing:", "\n")
comments_dtm
cat("\n")
cat("Our matrix without preprocessing:", "\n")
comments_dtm_no_prep

#Creating training(75%) and test(25%) datasets
comments_dtm_train <- comments_dtm[1:3693, ]
comments_dtm_test <- comments_dtm[3694:4923, ]
comments_train_labels <- mydata[1:3693, ]$label
comments_test_labels <- mydata[3694:4923, ]$label

cat("Our training data")
comments_train_labels %>%
  table %>%
  prop.table
cat("\n")
cat("Our testing data")
comments_test_labels %>%
  table %>%
  prop.table

wordcloud(comments_corpus_clean, min.freq = 20,colors = brewer.pal(8, 'Dark2'), random.order = FALSE)

#Bully and not bully cloud
bully <- mydata %>%
  subset(label == "bully")
bullyCloud <- wordcloud(bully$Comments, max.words = 300,colors = brewer.pal(8, 'Dark2'),rot.per=0.35, scale = c(3.5, 0.25), random.order = FALSE)
notbully <- mydata %>%
  subset(label == "not bully")
notbullyCloud <- wordcloud(notbully$Comments,max.words = 300,colors = brewer.pal(8, 'Dark2'),rot.per=0.35, scale = c(3, 0.5), random.order = FALSE)

#Feature Extraction
comments_dtm_freq_train <- comments_dtm_train %>%
  findFreqTerms(5) %>%
  comments_dtm_train[ , .]
comments_dtm_freq_test <- comments_dtm_test %>%
  findFreqTerms(5) %>%
  comments_dtm_test[ , .]

#From numeric to categorical "yes/no" matrices
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

comments_train <- comments_dtm_freq_train %>%
  apply(MARGIN = 2, convert_counts)
comments_test <- comments_dtm_freq_test %>%
  apply(MARGIN = 2, convert_counts)

#Training a model on the data
system.time(comments_classifier <- naiveBayes(comments_train, comments_train_labels))
system.time(comments_pred <- predict(comments_classifier, comments_test))


#Evaluating model performance
CrossTable(comments_pred, comments_test_labels, prop.chisq = FALSE, chisq = FALSE, 
           prop.t = FALSE,
           dnn = c("Predicted", "Actual"))

#Improving model performance
system.time(comments_classifier2 <- naiveBayes(comments_train, comments_train_labels, laplace = 1))
system.time(comments_pred2 <- predict(comments_classifier2, comments_test))
CrossTable(comments_pred2, comments_test_labels, prop.chisq = FALSE, chisq = FALSE, 
           prop.t = FALSE,
           dnn = c("Predicted", "Actual"))


# Prepare the confusion matrix
conf.mat <- confusionMatrix(comments_pred2,comments_test_labels)

conf.mat

conf.mat$byClass

conf.mat$overall

# Prediction Accuracy
conf.mat$overall['Accuracy']


