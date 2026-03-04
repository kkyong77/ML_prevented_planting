################################################################################
#
#                 CROP CHOICE PREDICTION MODEL USING RANDOM FOREST
#
# This script develops a machine learning model to predict crop choices for
# agricultural fields. The primary feature is the crop planted in the previous
# year.
#
# Workflow:
# 1.  Load and prepare historical crop data (2010-2019).
# 2.  Create a lagged feature for the previous year's crop.
# 3.  Train a Random Forest model on data from 2011-2017.
# 4.  Evaluate the model's performance on a 2018 validation set.
# 5.  Use the trained model to forecast crop choices for 2019.
# 6.  Visualize the model's performance and the 2019 forecast.
#
################################################################################


# --- 1. SETUP: LOAD LIBRARIES --- #########
 ------------------------------------------------------------------------------
# Load all necessary packages for the analysis.

library(dplyr)         # For data manipulation
library(tidyr)         # For data reshaping (gather)
library(readr)         # For reading/writing CSV files
library(randomForest)  # For the Random Forest model
library(caret)         # For performance evaluation (confusionMatrix)
library(fastDummies)   # For one-hot encoding categorical variables
library(sf)            # For handling spatial data (shapefiles)
library(ggplot2)       # For plotting
library(patchwork)     # For combining multiple plots


# --- 2. DATA PREPARATION: CREATING THE LAGGED FEATURE ---#######
------------------------------------------------------------------------------
# We load historical crop data, transform it from wide to long format, and
# create the key predictor variable: the crop grown in the previous year.

# Load historical crop data
crop_hist <- read_csv("data/acpf_welb_reclassfied.csv")

# Reshape data from wide to long format
# Each row will represent a single field in a single year
crop_long <- crop_hist %>%
  gather(key = "Year", value = "Crop_Type", maj10:maj19) %>%
  mutate(
    Year = as.integer(sub("maj", "", Year)) + 2000,
    Crop_Type = as.factor(Crop_Type)
  ) %>%
  select(FBndID, Year, Crop_Type, CropSumry)

# Create a lagged variable for the previous year's crop ('Crop_Type_Lag1')
# This is the primary feature for our model.
grid_data <- crop_long %>%
  group_by(FBndID) %>%
  arrange(Year) %>%
  mutate(Crop_Type_Lag1 = lag(Crop_Type, 1)) %>%
  ungroup() %>%
  filter(!is.na(Crop_Type_Lag1)) # Remove rows where the lag is NA (i.e., the first year, 2010)


# --- 3. MODEL TRAINING ---########
 ------------------------------------------------------------------------------
# We split the data into training (2011-2017) and validation (2018) sets,
# one-hot encode the categorical feature, and train the Random Forest model.

# Split dataset into training and validation sets based on year
train_data <- grid_data %>% filter(Year <= 2017)
valid_data <- grid_data %>% filter(Year == 2018)

# One-hot encode the lagged crop type feature
train_data_encoded <- dummy_cols(train_data, select_columns = "Crop_Type_Lag1", remove_selected_columns = TRUE)
valid_data_encoded <- dummy_cols(valid_data, select_columns = "Crop_Type_Lag1", remove_selected_columns = TRUE)

# Ensure the validation set has the same columns as the training set.
# This handles cases where a crop type might be present in the training lag
# but not in the validation lag.
missing_cols <- setdiff(names(train_data_encoded), names(valid_data_encoded))
if (length(missing_cols) > 0) {
  valid_data_encoded[missing_cols] <- 0
}
valid_data_encoded <- valid_data_encoded[names(train_data_encoded)] # Ensure same order

# Prepare feature matrices (X) and target vectors (y)
x_train <- train_data_encoded %>% select(-FBndID, -Year, -Crop_Type, -CropSumry)
y_train <- train_data_encoded$Crop_Type

x_valid <- valid_data_encoded %>% select(-FBndID, -Year, -Crop_Type, -CropSumry)
y_valid <- factor(valid_data_encoded$Crop_Type, levels = levels(y_train))

# Train the Random Forest model
set.seed(123) # for reproducibility
rf_crop_choice_model <- randomForest(
  x = x_train,
  y = y_train,
  ntree = 500,
  importance = TRUE
)

# Save the trained model object for future use
write_rds(rf_crop_choice_model, "data/crop_choice_model/rf_model3.rds")

rf_crop_choice_model<-read_rds("data/crop_choice_model/rf_model3.rds")


# --- 4. MODEL EVALUATION ---######
 ------------------------------------------------------------------------------
# We evaluate the model's accuracy on both the training and validation sets
# and visualize the results using confusion matrices and variable importance plots.

# --- Evaluate on Training Data ---
train_pred <- predict(rf_crop_choice_model, x_train)
train_cm <- confusionMatrix(train_pred, y_train)
train_accuracy <- train_cm$overall['Accuracy']
cat("Training Accuracy:", train_accuracy, "\n")

# --- Evaluate on Validation Data ---
valid_pred <- predict(rf_crop_choice_model, x_valid)
valid_cm <- confusionMatrix(valid_pred, y_valid)
valid_accuracy <- valid_cm$overall['Accuracy']
cat("Validation Accuracy:", valid_accuracy, "\n")

# --- Visualize Confusion Matrices ---
# Function to create a confusion matrix plot
create_cm_plot <- function(cm, title) {
  cm_df <- as.data.frame(cm$table)
  ggplot(cm_df, aes(Prediction, Reference, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = Freq), color = "black", size = 3) +
    scale_fill_gradient(low = "white", high = "#B2182B") +
    labs(title = title, x = "Predicted", y = "Actual") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

plot_train_cm <- create_cm_plot(train_cm, paste0("(a) Training Accuracy: ", round(train_accuracy, 3)))
plot_valid_cm <- create_cm_plot(valid_cm, paste0("(b) Validation Accuracy: ", round(valid_accuracy, 3)))

# Combine and save the plots
combined_cm_plot <- plot_train_cm + plot_valid_cm
ggsave("figures/figS2_crop_choice_model3_train_valid_plot.jpg", plot = combined_cm_plot, width = 12, height = 6, dpi = 300)

# --- Save Model Accuracy & Variable Importance ---
model_accuracy <- data.frame(train_accuracy = train_accuracy, valid_accuracy = valid_accuracy, model = "M3")
write_csv(model_accuracy, file = "data/crop_choice_model/m3_accuracy.csv")

# --- 5. FORECASTING FOR 2019 ---#########
 ------------------------------------------------------------------------------
# We use the trained model to predict the most likely crop choices for 2019.
# The actual crop grown in 2018 serves as the 'Crop_Type_Lag1' feature for this forecast.

# Prepare the 2019 forecast data
# The previous year's crop ('Crop_Type_Lag1') is the actual crop from 2018.
forecast_data_2019 <- grid_data %>%
  filter(Year == 2018) %>%
  mutate(
    Year = 2019,
    Crop_Type_Lag1 = Crop_Type # The 2018 crop becomes the lag for 2019
  )

# One-hot encode the feature, ensuring column consistency with the training data
forecast_data_encoded <- dummy_cols(forecast_data_2019, select_columns = "Crop_Type_Lag1", remove_selected_columns = TRUE)
missing_cols_forecast <- setdiff(names(x_train), names(forecast_data_encoded))
if (length(missing_cols_forecast) > 0) {
  forecast_data_encoded[missing_cols_forecast] <- 0
}
x_forecast_2019 <- forecast_data_encoded[names(x_train)] # Ensure same columns and order

# Predict probabilities for 2019
forecast_prob <- predict(rf_crop_choice_model, x_forecast_2019, type = "prob")

# Function to extract the top two choices and the max probability
get_top_choices <- function(prob_matrix) {
  top2 <- t(apply(prob_matrix, 1, function(row) {
    sorted_row <- sort(row, decreasing = TRUE)
    c(names(sorted_row)[1], names(sorted_row)[2], sorted_row[1])
  }))
  colnames(top2) <- c("First_Choice", "Second_Choice", "Max_Prob")
  return(as.data.frame(top2))
}

# Get the top choices and probabilities
top_choices_2019 <- get_top_choices(forecast_prob)
top_choices_2019$Max_Prob <- as.numeric(top_choices_2019$Max_Prob)

# Combine results into a final forecast dataframe
results_2019 <- forecast_data_2019 %>%
  select(FBndID, Year) %>%
  bind_cols(top_choices_2019)

# Save the non-spatial forecast results
write_csv(results_2019, "data/model3_results_welb.csv")


# --- 6. VISUALIZATION OF 2019 FORECAST ---####
# ------------------------------------------------------------------------------
# We merge the forecast results with spatial data to create maps of the
# predicted first choice, second choice, and the model's confidence.

# Load the field boundaries shapefile
acpf_welb_sf <- st_read("gis/welb_merged_acpf.shp")

# Join the 2019 forecast results to the spatial data
forecast_sf <- left_join(acpf_welb_sf, results_2019, by = "FBndID")

# Save the final spatial data with predictions
st_write(forecast_sf, "gis/model3_results_welb.shp", delete_layer = TRUE)

# Define color and label mappings for consistent plotting
crop_labels <- c("B" = "Soybean", "C" = "Corn", "W" = "Winter Wheat", "I" = "Fallow/Idle", "P" = "Pasture", "R" = "Other")
crop_colors <- c("B" = "green", "C" = "yellow", "W" = "#D2B48C", "I" = "brown", "P" = "purple", "R" = "gray")

# Plot 1: Map of the First Crop Choice for 2019
p_map_first <- ggplot(forecast_sf) +
  geom_sf(aes(fill = First_Choice), color = NA) +
  scale_fill_manual(values = crop_colors, labels = crop_labels, name = "First Choice") +
  labs(title = "(a) Predicted First Crop Choice for 2019") +
  theme_void()

# Plot 2: Map of the Prediction Probability
p_map_prob <- ggplot(forecast_sf) +
  geom_sf(aes(fill = Max_Prob), color = NA) +
  scale_fill_viridis_c(option = "magma", name = "Probability") +
  labs(title = "(b) Probability of First Choice") +
  theme_void()

# Plot 3: Map of the Second Crop Choice for 2019
p_map_second <- ggplot(forecast_sf) +
  geom_sf(aes(fill = Second_Choice), color = NA) +
  scale_fill_manual(values = crop_colors, labels = crop_labels, name = "Second Choice") +
  labs(title = "(c) Predicted Second Crop Choice for 2019") +
  theme_void()

# Combine all three maps into a single figure
combined_forecast_plot <- (p_map_first | p_map_prob) / p_map_second +
  plot_annotation(title = "Crop Choice Forecast for 2019")

# Save the combined plot
ggsave("figures/figS3_crop_choice_model3.jpg", plot = combined_forecast_plot, width = 12, height = 8, dpi = 300)

################################ END OF SCRIPT #################################