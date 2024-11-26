# -------------------------------------------------------------------
# Title: Pi Digit Analysis and Predictive Modeling with Varying Window Sizes
# Description: 
#   - Analyzes the distribution of digits in π.
#   - Performs a chi-squared test for uniformity.
#   - Attempts to model the next digit in π using various machine learning algorithms.
#   - Varies the number of previous digits used as features (window sizes: 10, 20, ..., 100).
#   - Aggregates and visualizes model performance across different window sizes.
# -------------------------------------------------------------------

# ---------------------------
# 1. Load Required Libraries
# ---------------------------
# Install missing packages if necessary
required_packages <- c(
  "Rmpfr", "ggplot2", "DOYPAColors", "randomForest", "caret", "e1071",
  "class", "nnet", "dplyr", "conflicted", "binom", "scales", "tidyr", "progress",
  "gghighlight"
)

installed_packages <- rownames(installed.packages())
for (pkg in required_packages) {
  if (!(pkg %in% installed_packages)) {
    install.packages(pkg, dependencies = TRUE)
  }
}

# Load libraries
library(Rmpfr)          # For high-precision arithmetic
library(ggplot2)        # For plotting
library(DOYPAColors)    # For color palettes
library(randomForest)   # For Random Forest modeling
library(caret)          # For data splitting and preprocessing
library(e1071)          # For Support Vector Machines
library(class)          # For k-Nearest Neighbors
library(nnet)           # For Neural Networks
library(dplyr)          # For data manipulation
library(conflicted)     # To manage function conflicts
library(binom)          # For binomial confidence intervals
library(scales)         # For scale functions in ggplot2
library(tidyr)          # For data reshaping
library(progress)       # For progress bars
library(gghighlight)

# Resolve any function conflicts, preferring base::apply and base::matrix
conflicted::conflicts_prefer(base::apply)
conflicted::conflicts_prefer(base::matrix)

# ---------------------------
# 2. Set Seed for Reproducibility
# ---------------------------
set.seed(314) # obviously

# ---------------------------
# 3. Pi Digit Extraction
# ---------------------------
extract_pi_digits <- function(digits = 10000) {
  bits <- ceiling(digits * log2(10)) + 100  # Adding extra bits for precision
  pi_mpfr <- Const("pi", prec = bits)
  pi_str <- format(pi_mpfr, digits = digits + 1)  # +1 to account for the decimal point
  pi_digits <- gsub("\\.", "", pi_str)  # Remove the decimal point
  digit_list <- strsplit(pi_digits, split = "")[[1]]
  return(digit_list)
}

digit_list <- extract_pi_digits(digits = 10000)

# Convert digit list to numeric vector for modeling
digit_vector <- as.numeric(digit_list)

# ---------------------------
# 4. Digit Frequency Analysis
# ---------------------------
# Count the frequency of each digit (0-9)
digit_counts <- table(digit_list)

# Ensure all digits 0-9 are represented
all_digits <- as.character(0:9)
digit_counts <- digit_counts[all_digits]
digit_counts[is.na(digit_counts)] <- 0

# Convert to DataFrame for analysis
digit_counts_df <- as.data.frame(digit_counts)
names(digit_counts_df) <- c("Digit", "Count")

# Calculate expected counts for uniform distribution
digit_counts_df$Expected <- 10000 / nrow(digit_counts_df)

# Create a plot for this
digit_counts_df %>%
  ggplot(., aes(x = Digit, y = Count)) +
  geom_col() +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = .5)
  ) +
  labs(y = "Frequency",
       title = "Digit Distribution of Pi",
       subtitle = "First 10,000 Digits")
  
# Perform Chi-Squared Test
chi_squared_test <- chisq.test(x = digit_counts_df$Count, p = rep(1 / 10, 10))

# Display Chi-Squared Test Results
cat("Chi-Squared Test Results:\n")
print(chi_squared_test)

if (chi_squared_test$p.value < 0.05) {
  cat("Result: Reject the null hypothesis. The digit distribution is not uniform.\n\n")
} else {
  cat("Result: Fail to reject the null hypothesis. No significant deviation from uniform distribution.\n\n")
}

# ---------------------------
# 5. Randomness Analysis
# ---------------------------
# Parameters for randomness trial
trials <- 5000
digits <- 10000

# Initialize vectors to store results
matches_pct <- numeric(trials)
running_mean <- numeric(trials)

# Progress bar for randomness analysis
pb_randomness <- progress_bar$new(
  format = "  Randomness Analysis [:bar] :percent eta: :eta",
  total = trials, clear = FALSE, width=60
)

# Perform trials to compare random digit generation with π digits
for (i in 1:trials) {
  random_list <- floor(runif(digits, 0, 10))  # Generate random digits between 0 and 9
  matches_pct[i] <- sum(random_list == digit_vector) / digits
  
  # Calculate running mean
  if (i == 1) {
    running_mean[i] <- matches_pct[i]
  } else {
    running_mean[i] <- mean(matches_pct[1:i])
  }
  
  pb_randomness$tick()
}

# Create DataFrames for plotting
running_mean_data <- data.frame(
  Trial = 1:trials,
  RunningMean = running_mean
)

matches_pct_data <- data.frame(
  Trial = 1:trials,
  MatchesPct = matches_pct
)

# Plot Running Mean of Match Percentages
ggplot(running_mean_data, aes(x = Trial, y = RunningMean)) +
  geom_line(color = doypa("astro")[1], linewidth = 1.5) +
  theme_minimal() +
  scale_y_continuous(labels = scales::label_percent(accuracy = 0.01),
                     limits = c(0.095, 0.105)) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    legend.position = "bottom"
  ) +
  labs(
    title = "Running Mean of Match Percentages Over Trials",
    x = "Trial Number",
    y = "Running Mean (%)"
  )

# Plot Histogram of Match Percentages
ggplot(matches_pct_data, aes(x = MatchesPct)) +
  geom_histogram(binwidth = 0.0005, fill = doypa("astro")[1], color = "black")  +
  theme_minimal(base_size = 14) +
  scale_x_continuous(labels = scales::percent_format()) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    legend.position = "bottom"
  ) +
  labs(
    title = "Histogram of Match Percentages",
    x = "Match Percentage",
    y = "Frequency of Trials"
  )

# ---------------------------
# 6. Predictive Modeling with Varying Window Sizes
# ---------------------------

# Define the range of window sizes
window_sizes <- seq(10, 90, by = 5)

# Initialize a list to store accuracy results for each window size
all_accuracy_results <- list()

# Define maximum allowable weights for the neural network
max_weights <- 1000  # Reduced from 1200 to 1000 to accommodate nnet limitations

# Function to calculate maximum hidden units based on window size
calculate_max_hidden <- function(window_size, output_classes = 10, max_weights = 1000) {
  # Let:
  # W1 = window_size * hidden_units (input to hidden)
  # B1 = hidden_units (biases for hidden layer)
  # W2 = hidden_units * output_classes (hidden to output)
  # B2 = output_classes (biases for output layer)
  # Total Weights = W1 + B1 + W2 + B2 <= max_weights
  #
  # Total Weights = hidden_units * (window_size + output_classes) + hidden_units + output_classes
  #                = hidden_units * (window_size + output_classes + 1) + output_classes
  #
  # Solve for hidden_units:
  # hidden_units <= (max_weights - output_classes) / (window_size + output_classes + 1)
  
  hidden_units <- floor((max_weights - output_classes) / (window_size + output_classes + 1))
  return(max(hidden_units, 1))  # Ensure at least 1 hidden unit
}

# Progress bar for predictive modeling
pb_modeling <- progress_bar$new(
  format = "  Predictive Modeling [:bar] :percent eta: :eta",
  total = length(window_sizes), clear = FALSE, width=60
)

# Iterate over each window size
for (window_size in window_sizes) {
  pb_modeling$tick()
  cat("\n\nProcessing window size:", window_size, "digits\n")
  
  # 6.1 Data Preparation for Modeling
  create_dataset <- function(digits, window_size) {
    n <- length(digits) - window_size
    features <- matrix(nrow = n, ncol = window_size)
    target <- vector(length = n)
    
    for (i in 1:n) {
      features[i, ] <- digits[i:(i + window_size - 1)]
      target[i] <- digits[i + window_size]
    }
    
    df <- as.data.frame(features)
    colnames(df) <- paste0("D", 1:window_size)
    df$Target <- as.factor(target)  # Convert target to factor for classification
    return(df)
  }
  
  dataset <- create_dataset(digit_vector, window_size)
  
  # 6.2 Split into Training and Testing Sets
  set.seed(123)  # For reproducibility
  train_index <- createDataPartition(dataset$Target, p = 0.8, list = FALSE)
  train_data <- dataset[train_index, ]
  test_data <- dataset[-train_index, ]
  
  # 6.3 Feature Scaling for Models that Require It
  preProcValues <- preProcess(train_data[, 1:window_size], method = c("center", "scale"))
  train_scaled <- predict(preProcValues, train_data[, 1:window_size])
  test_scaled <- predict(preProcValues, test_data[, 1:window_size])
  
  train_labels <- train_data$Target
  test_labels <- test_data$Target
  
  # 6.4 Train Various Models
  
  ## 6.4.1 Random Forest
  rf_model <- randomForest(Target ~ ., data = train_data, ntree = 100, importance = TRUE)
  
  ## 6.4.2 Multinomial Logistic Regression
  logistic_model <- multinom(Target ~ ., data = train_data, trace = FALSE)
  
  ## 6.4.3 k-Nearest Neighbors (k-NN)
  k <- 5  # Number of neighbors
  
  ## 6.4.4 Support Vector Machines (SVM)
  svm_model <- svm(Target ~ ., data = train_data, kernel = "linear", probability = TRUE)
  
  ## 6.4.5 Neural Networks
  # Calculate the maximum number of hidden units allowed
  max_hidden <- calculate_max_hidden(window_size, output_classes = 10, max_weights = max_weights)
  cat("  Calculated maximum hidden units for Neural Network:", max_hidden, "\n")
  
  if (max_hidden >= 1) {
    nn_model <- tryCatch(
      {
        nnet(Target ~ ., data = train_data, size = max_hidden, maxit = 200, decay = 5e-4, trace = FALSE)
      },
      error = function(e) {
        cat("  Skipping Neural Network due to error:", e$message, "\n")
        return(NULL)
      }
    )
  } else {
    cat("  Skipping Neural Network: Window size too large.\n")
    nn_model <- NULL
  }
  
  # 6.5 Evaluate Model Performance
  
  # Initialize DataFrame to store accuracies
  accuracy_results <- data.frame(
    Model = character(),
    Accuracy = numeric(),
    stringsAsFactors = FALSE
  )
  
  # 6.5.1 Random Forest Predictions
  rf_predictions <- predict(rf_model, newdata = test_data)
  rf_conf_matrix <- confusionMatrix(rf_predictions, test_data$Target)
  rf_accuracy <- rf_conf_matrix$overall['Accuracy']
  accuracy_results <- rbind(accuracy_results, data.frame(Model = "Random Forest", Accuracy = rf_accuracy))
  
  # 6.5.2 Multinomial Logistic Regression Predictions
  logistic_predictions <- predict(logistic_model, newdata = test_data)
  logistic_conf_matrix <- confusionMatrix(logistic_predictions, test_data$Target)
  logistic_accuracy <- logistic_conf_matrix$overall['Accuracy']
  accuracy_results <- rbind(accuracy_results, data.frame(Model = "Multinomial Logistic Regression", Accuracy = logistic_accuracy))
  
  # 6.5.3 k-Nearest Neighbors Predictions
  knn_predictions <- knn(train = train_scaled, test = test_scaled, cl = train_labels, k = k)
  knn_conf_matrix <- confusionMatrix(knn_predictions, test_labels)
  knn_accuracy <- knn_conf_matrix$overall['Accuracy']
  accuracy_results <- rbind(accuracy_results, data.frame(Model = paste("k-NN (k =", k, ")"), Accuracy = knn_accuracy))
  
  # 6.5.4 Support Vector Machines Predictions
  svm_predictions <- predict(svm_model, newdata = test_data)
  svm_conf_matrix <- confusionMatrix(svm_predictions, test_data$Target)
  svm_accuracy <- svm_conf_matrix$overall['Accuracy']
  accuracy_results <- rbind(accuracy_results, data.frame(Model = "Support Vector Machine", Accuracy = svm_accuracy))
  
  # 6.5.5 Neural Network Predictions
  if (!is.null(nn_model)) {
    nn_predictions_prob <- predict(nn_model, newdata = test_data[, 1:window_size], type = "raw")
    # nnet returns probabilities; assign class with highest probability
    nn_predictions <- apply(nn_predictions_prob, 1, which.max) - 1  # Adjust if classes start at 1
    nn_predictions <- as.factor(nn_predictions)
    # Ensure factor levels match
    nn_predictions <- factor(nn_predictions, levels = levels(test_data$Target))
    
    nn_conf_matrix <- confusionMatrix(nn_predictions, test_data$Target)
    nn_accuracy <- nn_conf_matrix$overall['Accuracy']
    accuracy_results <- rbind(accuracy_results, data.frame(Model = "Neural Network", Accuracy = nn_accuracy))
  } else {
    # Assign NA for accuracy if the model was skipped
    accuracy_results <- rbind(accuracy_results, data.frame(Model = "Neural Network", Accuracy = NA))
  }
  
  # 6.5.6 Random Guessing Baseline
  random_guesses <- sample(0:9, nrow(test_data), replace = TRUE) %>% factor(levels = levels(test_data$Target))
  random_conf_matrix <- confusionMatrix(random_guesses, test_data$Target)
  random_accuracy <- random_conf_matrix$overall['Accuracy']
  accuracy_results <- rbind(accuracy_results, data.frame(Model = "Random Guessing", Accuracy = random_accuracy))
  
  # 6.6 Calculate Confidence Intervals for Accuracies
  accuracy_results <- accuracy_results %>%
    rowwise() %>%
    mutate(
      Correct = ifelse(is.na(Accuracy), NA, round(Accuracy * length(test_labels))),
      Lower = ifelse(!is.na(Correct),
                     binom.confint(x = Correct, n = length(test_labels), methods = "wilson")$lower,
                     NA),
      Upper = ifelse(!is.na(Correct),
                     binom.confint(x = Correct, n = length(test_labels), methods = "wilson")$upper,
                     NA)
    ) %>%
    ungroup()
  
  # Add Window Size to Results
  accuracy_results$WindowSize <- window_size
  
  # Store the results in the list
  all_accuracy_results[[as.character(window_size)]] <- accuracy_results
}

# ---------------------------
# 7. Aggregating and Visualizing Results
# ---------------------------

# Combine all results into a single DataFrame
combined_accuracy <- bind_rows(all_accuracy_results, .id = "WindowSize")

# Convert WindowSize to numeric
combined_accuracy$WindowSize <- as.numeric(combined_accuracy$WindowSize)

# Display Combined Model Accuracies with Confidence Intervals
cat("\n\nCombined Model Accuracies with 95% Confidence Intervals across Window Sizes:\n")
print(combined_accuracy)

# 7.1 Visualization: Model Accuracies Across Window Sizes
# Load necessary libraries
library(ggplot2)
library(dplyr)

# Assuming 'combined_accuracy' is your data frame
# Calculate mean accuracy and confidence intervals for each model
model_summary <- combined_accuracy %>%
  group_by(Model) %>%
  summarise(
    MeanAccuracy = mean(Accuracy),
    MeanLower = mean(Lower),
    MeanUpper = mean(Upper)
  )

# Perform t-tests comparing each model to Random Guessing
# Create a vector to store p-values
model_summary$p_value <- NA

# Get the accuracies for Random Guessing
rg_accuracies <- combined_accuracy %>%
  filter(Model == "Random Guessing") %>%
  pull(Accuracy)

# Loop through each model to compute p-values
for (i in 1:nrow(model_summary)) {
  model_name <- model_summary$Model[i]
  
  if (model_name != "Random Guessing") {
    model_accuracies <- combined_accuracy %>%
      filter(Model == model_name) %>%
      pull(Accuracy)
    
    # Perform t-test
    ttest_result <- t.test(model_accuracies, rg_accuracies)
    model_summary$p_value[i] <- ttest_result$p.value
  }
}

# Determine significance levels
model_summary <- model_summary %>%
  mutate(
    significance = case_when(
      p_value < 0.001 ~ "***",
      p_value < 0.01 ~ "**",
      p_value < 0.05 ~ "*",
      TRUE ~ ""
    )
  )

# Extract the mean accuracy for Random Guessing
random_guessing_accuracy <- model_summary %>%
  filter(Model == "Random Guessing") %>%
  pull(MeanAccuracy)

# Create the plot
ggplot(model_summary, aes(x = reorder(Model, MeanAccuracy), y = MeanAccuracy, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6) +
  scale_fill_manual(values = c(doypa("astro")[1], doypa("astro")[1], doypa("astro")[1], doypa("astro")[1], doypa("astro")[2], doypa("astro")[1])) +
  scale_y_continuous(labels = scales::label_percent(accuracy = 1)) +
  geom_errorbar(aes(ymin = MeanLower, ymax = MeanUpper), width = 0.2, color = "darkgray") +
  geom_hline(yintercept = random_guessing_accuracy, linetype = "dashed", color = doypa("astro")[2]) +
  coord_flip() +
  theme_minimal(base_size = 14) +
  labs(
    title = "Model Mean Accuracies with Confidence Intervals",
    x = NULL,
    y = "Mean Accuracy"
  ) +
  geom_text(aes(label = significance), hjust = -0.2, vjust = 0.5, color = "black", size = 5) +
  theme(
    plot.title = element_text(hjust = 0.5),
    axis.title.y = element_blank()
  ) +
  guides(fill = "none")

ggplot(combined_accuracy, aes(x = Model, y = Accuracy)) +
  geom_point() +
  scale_y_continuous(labels = scales::label_percent(accuracy = 1)) +
  theme_minimal(base_size = 14) +
  coord_flip() +
  gghighlight(Accuracy > .11, label_key = WindowSize)+
  labs(
    title = "Model Accuracies Across Window Sizes",
    x = NULL,
    y = "Accuracy"
  )+
  theme(
    plot.title = element_text(hjust = 0.5),
    axis.title.y = element_blank()
  )





