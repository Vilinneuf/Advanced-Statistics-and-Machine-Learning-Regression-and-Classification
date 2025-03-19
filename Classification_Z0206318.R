# ==========================
# Part 1: Data Exploration
# ==========================
# Load necessary libraries
library(ggplot2)
library(dplyr)
library(caret)
library(randomForest)
library(xgboost)
library(Matrix)
library(reshape2)
library(pROC)

# Read data
data_url <- "https://www.louisaslett.com/Courses/MISCADA/hotels.csv"
df <- read.csv(data_url, header = TRUE)

# Remove variables that may cause data leakage
df <- df %>% select(-c(reservation_status, reservation_status_date))

# Convert target variable
df$is_canceled <- as.factor(df$is_canceled)

# Data summary and preview of first few rows
print("Data Summary:")
print(summary(df))
print("First few rows of data:")
print(head(df))

# Histograms for numeric variables
numeric_cols <- sapply(df, is.numeric)
df_numeric <- df[, numeric_cols]
df_melt <- melt(df_numeric)
print(ggplot(df_melt, aes(x = value)) +
        geom_histogram(bins = 30, fill = "blue", color = "black") +
        facet_wrap(~variable, scales = "free") +
        labs(title = "Histogram of Numeric Variables"))

# Bar plots for categorical variables
cat_cols <- sapply(df, function(x) is.factor(x) || is.character(x))
for(col in names(df)[cat_cols]) {
  print(ggplot(df, aes_string(x = col)) +
          geom_bar(fill = "orange", color = "black") +
          labs(title = paste("Bar Plot of", col)))
}

# ==========================
# Part 2: Data Preprocessing
# ==========================
# Split data into training and test sets (avoid data leakage)
set.seed(1234)
split <- sample(nrow(df), size = 0.8 * nrow(df))
trainData <- df[split, ]
testData <- df[-split, ]

# Fill missing values
for (col in names(trainData)) {
  if (is.numeric(trainData[[col]])) {
    trainData[[col]][is.na(trainData[[col]])] <- median(trainData[[col]], na.rm = TRUE)
    testData[[col]][is.na(testData[[col]])] <- median(testData[[col]], na.rm = TRUE)
  } else {
    trainData[[col]][is.na(trainData[[col]])] <- "Unknown"
    testData[[col]][is.na(testData[[col]])] <- "Unknown"
  }
}

# Ensure 'is_canceled' is a factor
trainData$is_canceled <- as.factor(trainData$is_canceled)
testData$is_canceled <- as.factor(testData$is_canceled)

# ==========================
# Part 3: Model Training & Evaluation - Default Models
# ==========================
# Train Random Forest (default parameters)
rf_model <- randomForest(is_canceled ~ ., data = trainData, ntree = 50, mtry = 3, nodesize = 5)
rf_pred <- predict(rf_model, testData, type = "class")
rf_acc <- confusionMatrix(rf_pred, testData$is_canceled)$overall["Accuracy"]

# Train XGBoost (default parameters)
formula <- as.formula(is_canceled ~ . -1)
train_matrix <- sparse.model.matrix(formula, data = trainData)
test_matrix <- sparse.model.matrix(formula, data = testData)

# Ensure that train_matrix and test_matrix have the same feature columns
common_cols <- intersect(colnames(train_matrix), colnames(test_matrix))
train_matrix <- train_matrix[, common_cols, drop = FALSE]
test_matrix <- test_matrix[, common_cols, drop = FALSE]

y_train <- as.numeric(as.character(trainData$is_canceled))
y_test <- as.numeric(as.character(testData$is_canceled))

xgb_model <- xgboost(data = train_matrix, label = y_train, max_depth = 4,
                     eta = 0.1, nrounds = 50, objective = "binary:logistic", verbose = 0)

xgb_pred <- as.numeric(predict(xgb_model, test_matrix) > 0.5)  # Convert predictions to 0/1
xgb_acc <- confusionMatrix(factor(xgb_pred, levels = c(0,1)), 
                           factor(y_test, levels = c(0,1)))$overall["Accuracy"]

# Comparison plot (Random Forest vs. XGBoost)
result_df <- data.frame(Model = c("Random Forest", "XGBoost"), Accuracy = c(rf_acc, xgb_acc))
print(ggplot(result_df, aes(x = Model, y = Accuracy, fill = Model)) +
        geom_bar(stat = "identity") +
        geom_text(aes(label = round(Accuracy, 4)), vjust = -0.5) +
        labs(title = "Random Forest vs. XGBoost"))

print(paste("Random Forest Accuracy:", round(rf_acc, 4)))
print(paste("XGBoost Accuracy:", round(xgb_acc, 4)))

# Additional evaluation metrics: Precision, Recall, AUC-ROC
# ---- Random Forest Evaluation ----
rf_probs <- predict(rf_model, testData, type = "prob")[,2]
rf_cm <- confusionMatrix(rf_pred, testData$is_canceled)
rf_precision <- rf_cm$byClass["Pos Pred Value"]
rf_recall <- rf_cm$byClass["Sensitivity"]
rf_roc <- roc(as.numeric(as.character(testData$is_canceled)), rf_probs)
rf_auc <- auc(rf_roc)

# ---- XGBoost (Default) Evaluation ----
xgb_cm <- confusionMatrix(factor(xgb_pred, levels = c(0,1)), 
                          factor(y_test, levels = c(0,1)))
xgb_precision <- xgb_cm$byClass["Pos Pred Value"]
xgb_recall <- xgb_cm$byClass["Sensitivity"]
xgb_probs <- predict(xgb_model, test_matrix)
xgb_roc <- roc(y_test, xgb_probs)
xgb_auc <- auc(xgb_roc)

print(paste("Random Forest - Precision:", round(rf_precision, 4), 
            "Recall:", round(rf_recall, 4), "AUC:", round(rf_auc, 4)))
print(paste("XGBoost (Default) - Precision:", round(xgb_precision, 4), 
            "Recall:", round(xgb_recall, 4), "AUC:", round(xgb_auc, 4)))

# ==========================
# Part 4: XGBoost Model Optimization
# ==========================
# Define hyperparameter search space
param_grid <- expand.grid(max_depth = c(3, 5, 7), eta = c(0.01, 0.1, 0.3), nrounds = c(50, 100))
opt_results <- data.frame()

# Grid search
for (i in 1:nrow(param_grid)) {
  set.seed(1234)
  model <- xgboost(data = train_matrix, label = y_train, 
                   max_depth = param_grid$max_depth[i],
                   eta = param_grid$eta[i], 
                   nrounds = param_grid$nrounds[i], 
                   objective = "binary:logistic", verbose = 0)
  
  # Predict on test set and calculate accuracy
  pred <- as.numeric(predict(model, test_matrix) > 0.5)
  acc <- confusionMatrix(factor(pred, levels = c(0,1)), 
                         factor(y_test, levels = c(0,1)))$overall["Accuracy"]
  
  print(paste("max_depth:", param_grid$max_depth[i], "eta:", param_grid$eta[i], 
              "nrounds:", param_grid$nrounds[i], "Acc:", acc))
  
  opt_results <- rbind(opt_results, c(param_grid$max_depth[i], param_grid$eta[i], 
                                      param_grid$nrounds[i], acc))
}

# Organize search results
colnames(opt_results) <- c("Max Depth", "Eta", "Rounds", "Accuracy")
print(opt_results)

# Select best parameters
best_index <- which.max(opt_results$Accuracy)
best_params <- as.numeric(opt_results[best_index, ])

# Train the final optimized XGBoost model
final_xgb_model <- xgboost(data = train_matrix, label = y_train, 
                           max_depth = best_params[1], 
                           eta = best_params[2], 
                           nrounds = best_params[3], 
                           objective = "binary:logistic", verbose = 0)

# Calculate the prediction accuracy of the optimized model
final_xgb_pred <- as.numeric(predict(final_xgb_model, test_matrix) > 0.5)
final_xgb_acc <- confusionMatrix(factor(final_xgb_pred, levels = c(0,1)), 
                                 factor(y_test, levels = c(0,1)))$overall["Accuracy"]

# Final model comparison
comparison_df <- data.frame(Model = c("Random Forest", "XGBoost (Default)", "XGBoost (Optimized)"), 
                            Accuracy = c(rf_acc, xgb_acc, final_xgb_acc))
print(ggplot(comparison_df, aes(x = Model, y = Accuracy, fill = Model)) +
        geom_bar(stat = "identity") +
        geom_text(aes(label = round(Accuracy, 4)), vjust = -0.5) +
        labs(title = "Final Model Comparison"))
print(paste("Optimized XGBoost Accuracy:", round(final_xgb_acc, 4)))

# Misclassification impact analysis (using final optimized XGBoost as example)
final_xgb_cm <- confusionMatrix(factor(final_xgb_pred, levels = c(0,1)), 
                                factor(y_test, levels = c(0,1)))
final_table <- final_xgb_cm$table
TN <- final_table[1,1]
FP <- final_table[1,2]
FN <- final_table[2,1]
TP <- final_table[2,2]

print("Final Optimized XGBoost Misclassification Analysis:")
print(paste("True Negatives:", TN, 
            "False Positives:", FP, 
            "False Negatives:", FN, 
            "True Positives:", TP))

misclass_df <- data.frame(Misclassification = c("False Positives", "False Negatives"),
                          Count = c(FP, FN))
print(ggplot(misclass_df, aes(x = Misclassification, y = Count, fill = Misclassification)) +
        geom_bar(stat = "identity") +
        geom_text(aes(label = Count), vjust = -0.5) +
        labs(title = "Misclassification Analysis for Final Optimized XGBoost"))

# Calculate ROC curve for the optimized XGBoost model
final_xgb_probs <- predict(final_xgb_model, test_matrix)
final_xgb_roc <- roc(y_test, final_xgb_probs)
final_xgb_auc <- auc(final_xgb_roc)

# Plot ROC curves for three models
plot(rf_roc, col="blue", lwd=2, main="ROC Curves for Random Forest & XGBoost")
plot(xgb_roc, col="red", lwd=2, add=TRUE)
plot(final_xgb_roc, col="green", lwd=2, add=TRUE)  # Green represents optimized XGBoost

# Add legend
legend("bottomright", legend=c("Random Forest", "XGBoost (Default)", "XGBoost (Optimized)"), 
       col=c("blue", "red", "green"), lwd=2)

# Construct data frame for metrics
metrics_df <- data.frame(
  Model = rep(c("Random Forest", "XGBoost"), each = 3),
  Metric = rep(c("Precision", "Recall", "AUC"), times = 2),
  Value = c(rf_precision, rf_recall, rf_auc, 
            xgb_precision, xgb_recall, xgb_auc)
)

# Plot bar chart for Precision, Recall, and AUC
ggplot(metrics_df, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = round(Value, 4)), vjust = -0.5, position = position_dodge(0.9)) +
  labs(title = "Comparison of Precision, Recall, and AUC",
       x = "Metric",
       y = "Value") +
  theme_minimal()

# Create AUC scores data frame
auc_df <- data.frame(
  Model = c("Random Forest", "XGBoost (Default)", "XGBoost (Optimized)"),
  AUC = c(rf_auc, xgb_auc, final_xgb_auc)
)

# Plot AUC comparison bar chart
ggplot(auc_df, aes(x = Model, y = AUC, fill = Model)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = round(AUC, 4)), vjust = -0.5) +
  labs(title = "AUC Comparison of Different Models",
       x = "Model",
       y = "AUC Value") +
  theme_minimal()
