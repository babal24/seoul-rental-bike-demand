#################################################
#Advanced Analytics & Machine Learning
#MGT7179
#Assignment 1 - Regression
#Vasileios Gounaris Bampaletsos-40314803

#regression problem
#bike-sharing dataset
#seoul's data for bike-sharing
#analyze and predict the demand of bikes per hour

#methods which used:
#linear regression (with cross validation)
#support vector machine (for regression)
#ridge  (with grid search)
#lasso  (with cross validation)
#gradient boosted machine x2 (1normal and 1tuned)
#boosted tress (xgboost method & cross validation)
###################################################


#set the working directory
setwd("/Users/basilisgounarismpampaletsos/Desktop/SEMESTER 2/PROJECTS 2/26_03 analytics")
options(scipen = 9)

#load the libraries
library(readxl)
library(psych)
library(ggplot2)
library(caTools)
library(statsr)
library(dplyr)
library(BAS)
library(car)
library(tidyr)
library(purrr)
library(gridExtra)
library(forcats)
library(corrplot)
library(magrittr)
library(caret)
library(Hmisc)
library(tidyverse)
library(ggpubr)
library(ROCR)
library(broom)
library(lubridate)
library(GGally)
library(ISLR)
library(hrbrthemes)
library(viridis)
library(e1071)
library(plyr)
library(readr)
library(repr)
library(glmnet)
library(ggthemes)
library(scales)
library(wesanderson)
library(styler)
library(xgboost)
library(randomForest)
library(rsample)      
library(gbm)          
library(h2o)          
library(pdp)          
library(lime)


#load the data
#dataset from Seoul's bike-sharing
data <- read.csv("SeoulBikeData.csv")

###########################################################
#summarize the data for the first time
#check the data's distribution and descriptive measures
summary(data)

#check some basic statistics for seoul's weather data
#looking for the minimum, median and maximum values
#understand the data's distribution
#looking for outliers
summary(data$rented_bike_count) #stats for number of the bikes
summary(data$temperature) #stats for temperature (Celcius)
summary(data$wind_speed) #stats for wind speed (miles per second (m/s))
summary(data$rainfall) #stats for rainfall (milimeters (mm))
summary(data$visibility) #stats for visibility (multipled by 10 meters)
summary(data$humidity) #stats for humidity (%)
summary(data$solar_radiation) #stats for solar radiation (MJ/m^2)
summary(data$snowfall) #stats for snowfall (cm)

aggregate(temperature ~ hour, data = data, FUN = mean)
aggregate(solar_radiation ~ hour, data = data, FUN = mean)


describeBy(data$rented_bike_count,data$hour)
describeBy(data$temperature, data$hour)

#make basic visualisations for better understanding
#make simple histograms for numerice variables
#histograms help to check the data quality

#NUMERIC VARIABLES
hist(data$rented_bike_count)
hist(data$rainfall)
hist(data$humidity)
hist(data$temperature)
hist(data$dew_point_temperature)
hist(data$wind_speed)
hist(data$snowfall)
hist(data$hour)
hist(data$visibility)
hist(data$solar_radiation)

#CATEGORICAL VARIABLES
#barplot with seasons and rented bikes
#see how many bikes are rented per season
ggplot(data, aes(seasons)) +
  geom_bar(colour="black", mapping = aes(fill = rented_bike_count)) +
  labs(fill="rented_bike_count", x="Seasons", y= "rented bike count", title="Rented Bike Count per Season")

#barplot with functioning day and rented bikes
#see how many bikes are rented if the day was functioning or not
ggplot(data, aes(functioning_day)) +
  geom_bar(colour="black", mapping = aes(fill = rented_bike_count)) +
  labs(fill="rented_bikes", x="functioning_day", y= "count", 
       title="Rented Bikes Distribution in Functioning days")

#barplot with holidy and rented bikes
#see how many bikes are rented if was holiday or not
ggplot(data, aes(holiday)) +
  geom_bar(colour="black", mapping = aes(fill = rented_bike_count)) +
  labs(fill="rented_bikes", x="holidays", y= "count", 
       title="Rented Bikes Distribution in Holidays")

##################################################################################################
#FIX THE DATA
##################################################################################################
#transform the type of the date to extract month
data$date <- as.Date(data$date, format="%d/%m/%Y")

#extract month from date variable
#it is a very nice variable(month) for the models
data$month <- months(data$date)

#my code was in greek language
#it gave me the months in greek language
#i transform the greek to english
data$month[data$month == "Ιανουαρίου"] <- "January" 
data$month[data$month == "Φεβρουαρίου"] <- "February" 
data$month[data$month == "Μαρτίου"] <- "March" 
data$month[data$month == "Απριλίου"] <- "April" 
data$month[data$month == "Μαΐου"] <- "May" 
data$month[data$month == "Ιουνίου"] <- "June" 
data$month[data$month == "Ιουλίου"] <- "July" 
data$month[data$month == "Αυγούστου"] <- "August" 
data$month[data$month == "Σεπτεμβρίου"] <- "September" 
data$month[data$month == "Οκτωβρίου"] <- "October" 
data$month[data$month == "Νοεμβρίου"] <- "November" 
data$month[data$month == "Δεκεμβρίου"] <- "December" 

#put the months to chronological serie
data$month <- factor(data$month, levels = c("January", "February", "March", "April", "May", 
                                            "June", "July", "August", 
                                            "September", "October", "November", "December"))
#delete the date valiable
data$date <- NULL
#data$humidity[data$humidity <10] <- NA
#data <- na.omit(data)

##################################################################################################
#FINAL VISUALISATIONS
#VISUALISE THE DATA FOR BETTER UNDERSTANDING
##################################################################################################
#vis 1
#check the impact of wind speed in bike rental
ggplot(data, aes(x=wind_speed, y=rented_bike_count, color=holiday)) + 
  geom_point(size=0.2) +
  labs(title = "The Impact of Wind in Bike Demand", x = "wind speed (m/s)", y = "rented bike count")

#vis 2
#check the distribution of bike rental from the eyes of visibility
ggplot(data, aes(x = visibility)) +
  geom_density(aes(y = ..count..), fill = "lightgray") +
  geom_vline(aes(xintercept = mean(visibility)), linetype = "dashed", size = 0.6, color = "#FC4E07") +
  labs(fill="visibility", x="duration", y= "Visibility (10m)", title="visibility distribution", 
       caption="With the mean line in red")

#check the visibility's impact in bike rental
ggplot(data = data, aes(x = rented_bike_count, y = visibility)) + 
  geom_point() +
  geom_smooth(method = "lm") +
  labs(title = "The impact of Visalibity in Bike Demand", x = "rented bike count", y = "visibility (10m)")

#vis 3
#check the humidity's impact in bike rental
ggplot(data = data, aes(x = rented_bike_count, y = humidity)) + 
  geom_point() +
  geom_smooth(method = "lm") +
  labs(title = "The impact of Humidity in Bike Demand", x = "rented bike count", y = "humidity(%)")

#vis 4
#check how temperature impacts the bike rental
ggplot(data) + 
  geom_point(mapping = aes(x = rented_bike_count, y = temperature)) +
  labs(x="rented bike count", y= "temperatute (Celcius)", title="The impact of temperature to bike demand", 
       caption="Scatter plot to see the values distribution")

#vis 5
#create a boxplot with the distribution of rented bike throughout year
ggplot(data, aes(x=month, y=rented_bike_count, fill=seasons)) + 
  geom_boxplot(alpha=0.3) +
  theme(legend.position="none") +
  labs(x="Month", y= "Rented Bike Count", title="Rented Bike Destribution throughout Year per Season")

#vis 6
#bar plot with the average bike rental per hour
ggplot(data = data) + 
  geom_bar(mapping = aes(x=as.factor(hour), y = rented_bike_count), stat = "summary", y.mean = "mean") + 
  labs(title = "Average Rented Bikes per Hour", x="hour", y="AVG rented bike count")

#vis 7
#bike rental by holiday
ggplot(data) +
  geom_density(aes(x = rented_bike_count,
                   fill = seasons), 
               alpha = 0.1) +
  scale_fill_brewer(palette = "Dark2") +
  
  theme_fivethirtyeight() +
  theme(axis.title = element_text()) + 
  labs(title = "Bike Rental Density By Holiday",
       fill = "Holiday",
       x = "Bike Rentals",
       y = "Density")

#vis 8
#bike rental by temperature and season
ggplot(data, aes(y = rented_bike_count, 
             x = temperature, 
             color = seasons)) +
  geom_point(show.legend = FALSE) +
  geom_smooth(se = FALSE,
              show.legend = FALSE) +
  facet_grid(~seasons) +
  scale_color_brewer(palette = "Dark2") +
  theme_fivethirtyeight() +
  theme(axis.title = element_text()) +
  ylab("Bike Rentals") +
  xlab("Temperature (°C)") +
  ggtitle("Bike Rental By Temperature per Season")

#vis 9
#rented bikes by solar radiation
ggplot(data) +
  geom_point(aes(y = rented_bike_count, 
                 x = solar_radiation, 
                 color = functioning_day),
             show.legend = FALSE) +
  facet_grid(~functioning_day) +
  scale_color_brewer(palette = "Dark2") +
  theme_fivethirtyeight() +
  theme(axis.title = element_text()) +
  labs(x="Solar Radiation (MJ/m^2)", y= "Bike Rentals", title="Bike Rentals by Solar Radiation", 
       caption="Two graphs depends on Functioning Day")
  

#data$seasons=as.factor(data$seasons)
#data$month=as.factor(data$month)
#data$holiday=as.factor(data$holiday)
#data$functioning_day=as.factor(data$functioning_day)

#split the data
#train_set 75%
#test_set 25%
set.seed(123)
split = sample.split(data$rented_bike_count, SplitRatio = 0.75)
train_set = subset(data, split == TRUE)
test_set = subset(data, split == FALSE)



##################################################################################################
#CORRELATIONS
##################################################################################################
#at this point we want to see the correlations how the variables connect each other
#check this connections/associations because they are important for models' building

#correlations for numeric variables
#select the numeric variables to build the cor matrix
continuous_var <- select(data, rented_bike_count,hour,temperature,humidity,wind_speed,
                         visibility,dew_point_temperature,solar_radiation,rainfall)

#use 3 different ways to check the correlation
#very useful graphs
#full cor matrix, half cor matrix, pair plots

#correlation matrix
#using pearson method 
continuous_var.cor = cor(na.omit(continuous_var), method = "pearson")
corrplot(continuous_var.cor)

#half correlation matrix
#using pearson method 
ggcorr(na.omit(continuous_var), method = c("everything", "pearson"))

#pair plots 
#using pearson method 
#check the relation between the intependent and depends variables
pairs.panels(data, 
             method = "pearson", # correlation method
             hist.col = "#00AFBB",
             density = TRUE,  # show density plots
             ellipses = TRUE # show correlation ellipses
)

#correlations for categorical variables
#using pearson's chi-squared test
chisq.test(data$seasons, data$rented_bike_count)
chisq.test(data$holiday, data$rented_bike_count)
chisq.test(data$functioning_day, data$rented_bike_count)
chisq.test(data$month, data$rented_bike_count)

##################################################################################################
#create the model
#testing the variables and choose the best model
model = lm(formula = rented_bike_count ~.-dew_point_temperature,
            data = train_set)
summary(model)

#linear regression
#making some examples before final selection
lin_reg = lm(formula = model1,
              data = train_set)
summary(lin_reg)

##################################################################################################
#CHECK ASSUMPTIONS
##################################################################################################
#assumption 1
#the mean of residuals is zero
mean(model$residuals)

#assumption 2
#we check 2 assumpitions with this plot
#1.Homoscedasticity of residuals or equal variance
#2.Normality of residuals
par(mfrow=c(2,2))  # set 2 rows and 2 column plot layout
plot(model)

#assumption 3
#multicollinearity
vif(model)

#assumption 4
#influential cases
influence.measures(model)

#assumption 5
#independent residuals
durbinWatsonTest(model)


######################################################################################################
#LINEAR REGRESSION
######################################################################################################
#with cross validation
set.seed(123) 
train.control <- trainControl(method = "cv", number = 10)
# Train the model
model1 <- train(rented_bike_count ~.-dew_point_temperature, data = train_set, method = "lm",
               trControl = train.control)
# Summarize the results
print(model1)

# Make predictions on the test data
lin_pred <- predict(model1, newdata = test_set)
lin_pred 

#Compute RMSE, MAE and R^2
postResample(lin_pred, test_set$rented_bike_count)


######################################################################################################
#SVM REGRESSION
######################################################################################################
#create the svm regression model
svm_reg = svm(rented_bike_count ~.-dew_point_temperature, data=train_set)
print(svm_reg) #print the 

# Make predictions on the test data
svm_pred = predict(svm_reg, test_set)

#plot to compare the predictions with actual numbers
x=1:length(test_set$rented_bike_count)
plot(x, test_set$rented_bike_count, pch=13, col="red")
lines(x, svm_pred, lwd="1", col="blue")

#Compute RMSE, MAE and R^2
postResample(svm_pred, test_set$rented_bike_count)

#plot the stats and results from svm regression model
plot(data)
points(data$rented_bike_count, svm_pred, col="red", pch=13)


######################################################################################################
#RIDGE
######################################################################################################
data = na.omit(data)

#create a matrix to run the ridge
x = model.matrix(rented_bike_count ~.-dew_point_temperature, data)

#select the indepented variable
y = data %>%
  select(rented_bike_count)%>%
  unlist()%>%
  as.numeric()

#create the ridge regression using grid search
grid = 10^seq(10, -2, length = 100) #get the lambda sequence
ridge_reg = glmnet(x, y, alpha = 0, lambda = grid)
ridge_reg

#find and plot the coef
dim(coef(ridge_reg))
plot(ridge_reg)

#find the lambda, regression coef. and error
ridge_reg$lambda[50]
coef(ridge_reg)[,50]
sqrt(sum(coef(ridge_reg)[-1,50]^2))

predict(ridge_reg, s=50, type="coefficients")[1:20,]

x_train = model.matrix(rented_bike_count~.-dew_point_temperature, train_set)
x_test = model.matrix(rented_bike_count~.-dew_point_temperature, test_set)

y_train = train_set %>%
  select(rented_bike_count) %>%
  unlist() %>%
  as.numeric()

y_test = test_set %>%
  select(rented_bike_count) %>%
  unlist() %>%
  as.numeric()

#tune the model using the best values
ridge_reg = glmnet(x_train, y_train, alpha=0, lambda = grid, thresh = 1e-12)
# Make predictions on the test data
ridge_pred = predict(ridge_reg, s = 4, newx = x_test)

#Compute RMSE, MAE and R^2
postResample(ridge_pred, test_set$rented_bike_count)


######################################################################################################
#LASSO
######################################################################################################
#create the lasso regression with alpha 1
#using cross validation to build better model
lasso_reg = glmnet(x_train,
                   y_train,
                   alpha = 1,
                   lambda = grid)
lasso_reg
plot(lasso_reg)

set.seed(1)
cv_output = cv.glmnet(x_train, y_train, alpha = 1) # Fit lasso model on training data
plot(cv_output) # Draw plot of training MSE as a function of lambda
best_lambda = cv_output$lambda.min # Select lamda that minimizes training MSE

# Make predictions on the test data
lasso_pred = predict(lasso_reg, s = best_lambda , newx = x_test) # Use best lambda to predict test data

#Compute RMSE, MAE and R^2
postResample(lasso_pred, test_set$rented_bike_count)
mean((lasso_pred - y_test)^2) # Calculate test MSE

out = glmnet(x, y, alpha = 1, lambda = grid) # Fit lasso model on full dataset
lasso_coef = predict(out, type = "coefficients", s = best_lambda )# Display coefficients using lambda chosen by CV
lasso_coef

# Display only non-zero coefficients
lasso_coef[lasso_coef != 0]


######################################################################################################
#GRADIENT BOOSTED MACHINE
######################################################################################################
#GBM1
#create the gbm regression model
#cross validation using 5 steps
gbm1 <- gbm(
  formula = rented_bike_count ~ . -dew_point_temperature,
  distribution = "gaussian",
  data = train_set,
  n.trees = 10000,
  interaction.depth = 1,
  shrinkage = 0.001,
  cv.folds = 5,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)  

# print results
print(gbm1)

# plot loss function as a result of n trees added to the ensemble
gbm.perf(gbm1, method = "cv")

#influence bar chart
par(mar = c(5, 8, 1, 1))
summary(
  gbm1, 
  cBars = 10,
  method = relative.influence, # also can use permutation.test.gbm
  las = 2
)

#Make predictions on the test data
gbm1_pred <- predict(gbm1, n.trees = gbm1$n.trees, test_set)
#Compute RMSE, MAE and R^2
postResample(gbm1_pred, test_set$rented_bike_count)

#GBM2
#tuning the GBM1
#making a second gbm to see the difference
#the target is to boost the gbm for better results

# train GBM model
gbm2 <- gbm(
  formula = rented_bike_count~.-dew_point_temperature,
  distribution = "gaussian",
  data = train_set,
  n.trees = 10000,
  interaction.depth = 3,
  shrinkage = 0.1,
  cv.folds = 5,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)  

# print results
print(gbm2)

# find index for n trees with minimum CV error
min_MSE <- which.min(gbm2$cv.error)
summary(gbm2$cv.error)

# plot loss function as a result of n trees added to the ensemble
gbm.perf(gbm2, method = "cv")

#influence bar chart
par(mar = c(5, 8, 1, 1))
summary(
  gbm2, 
  cBars = 10,
  method = relative.influence, # also can use permutation.test.gbm
  las = 2
)

# Make predictions on the test data
gbm2_pred <- predict(gbm2, n.trees = gbm2$n.trees, test_set)
#Compute RMSE, MAE and R^2
postResample(gbm2_pred, test_set$rented_bike_count)


######################################################################################################
#BOOSTED TREES (XGBoost)
######################################################################################################
# Fit the model on the training set
#using the XGBoost method and cross validation
set.seed(123)
model <- train(
  rented_bike_count ~.-dew_point_temperature, data = train_set, method = "xgbTree",
  trControl = trainControl("cv", number = 10)
)

# Best tuning parameter mtry
model$bestTune
# Make predictions on the test data
predictions <- model %>% predict(test_set)
head(predictions)
#Compute RMSE, MAE and R^2
postResample(predictions, test_set$rented_bike_count)

