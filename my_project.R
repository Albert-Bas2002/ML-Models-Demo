library(dplyr) 
library(ggplot2)
library(reshape2)
library(randomForest)
library(caret)
library(e1071)
library(pROC)


data <- read.csv("C:\\Users\\0_0\\Desktop\\Уник\\Dataset_heart_deseases.csv")
data <- subset(data, select = -X)

empty_string_nan_counts <- colSums(data == "" | is.na(data))
print(empty_string_nan_counts)

data <- subset(data, select = -slope)
data <- subset(data, select = -ca)
data <- subset(data, select = -thal)

column_names_bool_char <- list('sex','pain_type','fbs','restecg','exang','slope','thal')
column_names_numerical = c('age','trestbps','chol','thalch','oldpeak','ca')

column_names_bool_char <- list('sex','pain_type','fbs','restecg','exang')

column_names_numerical = c('age','trestbps','chol','thalch','oldpeak')


column_names_bool_char_c = c('sex','pain_type','fbs','restecg','exang')

for (column_name in column_names_bool_char) {
  mode_value <- names(sort(table(data[[column_name]]), decreasing = TRUE)[1])
  if ( mode_value==''){mode_value=names(sort(table(data[[column_name]]), decreasing = TRUE)[2])}
  data[[column_name]] <- ifelse(is.na(data[[column_name]]) | data[[column_name]] == "", mode_value, data[[column_name]])
}

data[column_names_numerical] <- data.frame(apply(data[column_names_numerical], 2, function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x)))
data<-unique(data)
data$num_bi <- ifelse(data$num == 0, 0, 1)


model <- glm(num_bi ~ sex + age +trestbps+ chol+fbs+restecg+exang+oldpeak+pain_type+thalch , data = data, family = "binomial")
summary(model)
threshold <- 0.5
data$predicted <- predict(model, type = "response")
data$predicted_class <- ifelse(data$predicted > threshold, 1, 0)

accuracy <- sum(data$predicted_class  == data$num_bi) / length(data$predicted_class)
print(paste("Точность предсказаний:", accuracy))

corr_matrix <- cor(data[column_names_numerical], method = "pearson")

# Ваш код для генерации цветовой палитры
my_color_palette <- scale_fill_gradient2(low = "yellow", mid = "white", high = "blue", midpoint = 0)

# Создание графика с использованием ggplot2
ggplot(data = melt(corr_matrix), aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  my_color_palette + 
  geom_text(aes(label = round(value, 2)), vjust = 1) +  # Добавление текста с коэффициентами
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggplot(data, aes(x = age, y = thalch)) +
  geom_point(color = "red") +
  labs(title = "Scatterplot",
       x = "age",
       y = "thalch") +
  geom_smooth(aes(color = "Line of Regression"), method = "lm", se = FALSE) +
  scale_color_manual(values = "blue", name='') +  
  theme(legend.position = c(0.87, 0.9),
        legend.text = element_text(size = 12),  # Adjust text size
        legend.title = element_text(size = 14),
        legend.background = element_rect(fill = "transparent", color = NA),
        panel.grid = element_blank())

ggplot(data, aes(x = age, y = trestbps)) +
  geom_point(color = "red") +
  labs(title = "Scatterplot",
       x = "age",
       y = "chol") +
  scale_color_manual(values = "blue", name='') +  
  theme(legend.position = c(0.87, 0.1),
        legend.text = element_text(size = 12),  # Adjust text size
        legend.title = element_text(size = 14),
        legend.background = element_rect(fill = "transparent", color = NA),
        panel.grid = element_blank())

numerical_columns <- data[column_names_numerical]


for (col_name in names(numerical_columns)) {
  binwidth_tmp=10
  if (col_name=='oldpeak'){
    binwidth_tmp=0.7
  }
  if (col_name=='chol'){
    binwidth_tmp=20
  }
  if (col_name=='age'){
    binwidth_tmp=1
  }

  p <- ggplot(data, aes(x = data[[col_name]])) +
    geom_histogram(binwidth =binwidth_tmp, fill = "blue", color = "black", alpha = 0.7) +
    labs(title = paste("Histogram of", col_name), x = col_name, y = "Frequency")+
    theme(panel.grid = element_blank())
  print(p)
}

for (col_name in names(data[column_names_bool_char_c])) {
  plot_data <- as.data.frame(table(data[, col_name]))
  
  p <- ggplot(plot_data, aes(x = Var1, y = Freq)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    labs(title = paste("Bar plot for", col_name),
         x = col_name,
         y = "Frequency")
  print(p)
}

Q1 <- quantile(data$chol, 0.25)
Q3 <- quantile(data$chol, 0.75)
IQR <- Q3 - Q1

outliers <- data$chol < (Q1 - 1.5 * IQR) | data$chol > (Q3 + 1.5 * IQR)

# Удаление выбросов
df_no_outliers <- data[!outliers, ]


ggplot(data, aes(x = sex, y = age)) +
  geom_boxplot(fill = "lightblue", color = "blue") +
  labs(title = "Influence of Gender on Serum Cholesterol") 


ggplot(df_no_outliers, aes(x = pain_type, y = chol)) +
  geom_boxplot(fill = "lightblue", color = "blue") +
  labs(title = "Influence of Gender on Serum Cholesterol") +
  annotate("text", x = 4, y = 400, 
           label = "Pr(>F) 0.018", 
           size = 4, color = 'blue')
model <- aov(chol~ pain_type, data=df_no_outliers)
summary(model)

ggplot(data, aes(x = pain_type, y = chol)) +
  geom_boxplot(fill = "lightblue", color = "blue") +
  labs(title = "Influence of Gender on Serum Cholesterol") +
  annotate("text", x = 3.7, y = 570, 
           label = "Pr(>F) 6.98e-05
           Without asymp Pr(>F) 0.43", 
           size = 4, color = 'blue')
  
model <- aov(chol~ pain_type, data=data)
summary(model)


ggplot(data, aes(x = sex, y = oldpeak)) +
  geom_boxplot(fill = "lightblue", color = "blue") +
  labs(title = "Influence of Gender on Serum Cholesterol")
small_olpeak_data <- data[data$oldpeak < 0, ]
colons_needed <- c("oldpeak", "sex", "num_bi", "fbs")
small_olpeak_data <- small_olpeak_data[, colons_needed, drop = FALSE]
print(small_olpeak_data)
data <- data[data$oldpeak > -0.01, ]
data <- data[data$trestbps > 50, ]
data$num_bi <- as.character(data$num_bi)
data$num <- as.character(data$num)


ggplot(data, aes(x =num_bi, y = chol)) +
  geom_boxplot(fill = "lightblue", color = "blue") +
  labs(title = "Influence of Gender on Serum Cholesterol")



plot_data <- data[data$chol < 201, ]
plot_data <- as.data.frame(table(plot_data[, 'num_bi']))

p <- ggplot(plot_data, aes(x = Var1, y = Freq)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = paste("Bar plot for", col_name),
       x = col_name,
       y = "Frequency")
print(p)

ggplot(data, aes(x = age, y = trestbps, fill = sex)) +
  geom_point(shape = 21, size = 4, alpha = 0.7) +
  labs(title = "Sexual differences") +
  theme_minimal()
ggplot(data, aes(x = age, y = thalch, fill = num_bi)) +
  geom_point(shape = 21, size = 4, alpha = 0.7) +
  labs(title = "Presence of a disease") +
  theme_minimal()
ggplot(data, aes(x = age, y = oldpeak, fill = num_bi)) +
  geom_point(shape = 21, size = 4, alpha = 0.7) +
  labs(title = "Presence of a disease") +
  theme_minimal()
result <- chisq.test(data$sex, data$num)
print(result)
result <- chisq.test(data$restecg, data$num)
print(result)
result <- chisq.test(data$exang, data$num)
print(result)
result <- chisq.test(data$fbs, data$num)
print(result)
data$num_bi <- as.numeric((data$num_bi))
  
  print("Точность предсказаний без выбросов chol  0.8062")
  print("Точность предсказаний без выбросов age  0.8071")
  print("Точность предсказаний без выбросов thaclh  0.8056")
  print("Точность предсказаний без  asymptomatic  0.796")

data_new=data
model <- glm(num_bi ~ sex + age + chol+fbs+restecg+exang+oldpeak+pain_type+thalch , data = data_new, family = "binomial")
summary(model)
print('убрал trestbps')
threshold <- 0.5
data_new$predicted <- predict(model, type = "response")
data_new$predicted_class <- ifelse(data_new$predicted > threshold, 1, 0)

accuracy <- sum(data_new$predicted_class  == data_new$num_bi) / length(data_new$predicted_class)
print(paste("Точность предсказаний:", accuracy))

splitIndex <- createDataPartition(data$num, p = 0.8, list = FALSE)
train_data <- data[splitIndex, ]
test_data <- data[-splitIndex, ]

rf_model <- randomForest(as.factor(num) ~ ., data = train_data, ntree = 5)
predictions_tree <- predict(rf_model, data, type = "response")
conf_matrix_tree <- table(predictions_tree, data$num)
print(conf_matrix_tree)
accuracy_tree <- sum(diag(conf_matrix_tree)) / sum(conf_matrix_tree)
print(paste("Forest: ", accuracy_tree))


rf_model <- naiveBayes(as.factor(num) ~ ., data = train_data)
predictions_nb <- predict(rf_model, data, type = "response")
print(predictions_nb)
conf_matrix_nb <- table(predictions_nb, data$num)
print(conf_matrix_nb)
accuracy_nb <- sum(diag(conf_matrix_nb)) / sum(conf_matrix_nb)
print(paste("Naive Bayes : ", accuracy_nb))

rf_model <-svm(as.factor(num) ~ ., data = train_data)
predictions_svm <- predict(rf_model, data, type = "response")
conf_matrix_svm <- table(predictions_svm, data$num)
print(conf_matrix_svm)
accuracy_svm <- sum(diag(conf_matrix_svm)) / sum(conf_matrix_svm)
print(paste("SVM Accuracy : ", accuracy_svm))

estimate_list <- c(1.229619, 0.026877,0.0029698, -0.003489, 0.566395, -0.080719, -0.081150, 1.002521, 0.629348, -2.108990, -1.460137, -1.305455, -0.012961)
estimate_list <- abs(estimate_list)
columns_list <- c('sexMale', 'age','trestbps', 'chol', 'fbsTRUE', 'restecgnormal', 'restecgst-t abnormality', 'exangTRUE', 'oldpeak', 'pain_typeatypical angina', 'pain_typenon-anginal', 'pain_typetypical angina', 'thalch')
df <- data.frame(variables = columns_list, estimates = estimate_list)

# Создание барплота
ggplot(df, aes(x = variables, y = estimates)) +
  geom_bar(stat = "identity", fill = "blue", alpha = 0.7) +
  labs(title = "Значимость переменных",
       x = "",
       y = "Оценки") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 16))

conf_matrix_df_model1 <- as.data.frame(as.table(conf_matrix_nb))
conf_matrix_df_model1 <- conf_matrix_df_model1 %>%
  rename(Var1 = predictions_nb)# Для второй модели
conf_matrix_df_model2 <- as.data.frame(as.table(conf_matrix_svm))
conf_matrix_df_model2 <- conf_matrix_df_model2 %>%
  rename(Var1 = predictions_svm)# Для второй модели
# Для третьей модели
conf_matrix_df_model3 <- as.data.frame(as.table(conf_matrix_tree))
conf_matrix_df_model3 <- conf_matrix_df_model3 %>%
  rename(Var1 = predictions_tree)# Для второй модели
# Объединение данных в один датафрейм с указанием модели
conf_matrix_df_model1$Model <- "naiveBayes"
conf_matrix_df_model2$Model <- "SVM"
conf_matrix_df_model3$Model <- "Random_Forest"
conf_matrix_df_concat <- bind_rows(conf_matrix_df_model1, conf_matrix_df_model2,conf_matrix_df_model3)


# Объединение данных

# Настройка тепловой карты
heat_map_combined <- ggplot(conf_matrix_df_concat, aes(x = Var1, y = Var2, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = ifelse(Freq >= 10 & Freq <= 400, as.character(Freq), "")), vjust = 1) +  # Показывать значения только для 25 <= Freq <= 400
  scale_fill_gradient(low = "white", high = "blue") +
  theme_minimal() +
  facet_wrap(~Model) +  # Разделение по моделям
  labs(title = "Confusion Matrix Heatmap for models",
       x = "Actual Class stage of disease level",
       y = "Predicted Class stage of disease level",
       fill = "Frequency")

# Вывод тепловой карты
print(heat_map_combined)

