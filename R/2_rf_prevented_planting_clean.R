################################################################################
#
#       RANDOM FOREST MODEL FOR PREDICTING PREVENTED PLANTING
#
# This script builds and interprets a Random Forest model to predict whether a
# farm field will be prevented from planting ('P') or implemented ('I').
#
# Workflow:
# 1.  Load and prepare the data, including feature selection.
# 2.  Impute missing values using the 'missRanger' algorithm.
# 3.  Split the data into training and validation sets.
# 4.  Train a Random Forest classifier.
# 5.  Evaluate model performance on both training and validation data.
# 6.  Analyze feature importance using both standard RF metrics and SHAP values.
# 7.  Visualize the marginal effects of top features using ALE plots.
#
################################################################################


# --- 1. SETUP: LOAD LIBRARIES ---#######
 ------------------------------------------------------------------------------
# Load all necessary packages for the analysis.

library(dplyr)         # For data manipulation
library(readr)         # For reading CSV files
library(randomForest)  # For the Random Forest model
library(missRanger)    # For imputing missing data
library(caret)         # For data splitting and performance evaluation
library(fastshap)      # For calculating SHAP values
library(iml)           # For model interpretation (ALE plots)
library(ggplot2)       # For plotting
library(patchwork)     # For combining multiple plots
library(doParallel)    # For parallel processing


# --- 2. DATA PREPARATION AND IMPUTATION ---
 ------------------------------------------------------------------------------
# Load data, define variable types, and impute missing values.

# Load the primary dataset
data <- read_csv("data/rf_inputs/rf_inputs_0521.csv")

# Select relevant features and convert character/numeric types to factors
data <- data %>%
  select(
    "FBndID", "obs", "Acres", "Ksat", "clay", "men_RCI", "Max_Prob",
    "First_Choice", "Second_Choice", "seetile", "acc_d8", "wslope", "elev",
    "wet", "nccpi_all", "pr_m3", "pr_m4", "pr_m5", "pr_m6", "deficit_m3",
    "deficit_m4", "deficit_m5", "deficit_m6", "FERT_ACRES", "INCOME_PER_OP",
    "PCT_FARMING", "median_acres_op", "PCT_RES", "PCT_RENTED", "Farmer_ages",
    "MACH_ACRES", "CHEM_ACRES", "LAB_ACRES", "FED_ACRES", "PER_INSUR"
  ) %>%
  mutate(
    First_Choice = as.factor(First_Choice),
    Second_Choice = as.factor(Second_Choice),
    obs = factor(obs, levels = c("P", "I")), # P = Prevented, I = Implemented
    seetile = ifelse(is.na(seetile), 0, seetile) # Assume NA for tile means no tile
  )

# Impute remaining missing values using missRanger.
# This uses a random forest approach to predict and fill NAs.
# We exclude FBndID as it is an identifier, not a predictor.
set.seed(123) # for reproducible imputation
imputed_data <- missRanger(data, . ~ . - FBndID, pmm.k = 5, num.trees = 100)

# Save the imputed dataset
write_rds(imputed_data, "data/rf_model/imputed_data_0521.rds")


# --- 3. DATA SPLITTING ---
 ------------------------------------------------------------------------------
# Split the data into 80% for training and 20% for validation.
# A stratified split is used to maintain the proportion of P/I in both sets.

imputed_data <- read_rds("data/rf_model/imputed_data_0521.rds")

set.seed(123) # for reproducible splitting
train_index <- createDataPartition(imputed_data$obs, p = 0.8, list = FALSE)

train_data <- imputed_data[train_index, ]
val_data   <- imputed_data[-train_index, ]

# Prepare data for RF model (remove identifier column)
train_data_rf <- train_data %>% select(-FBndID)
val_data_rf   <- val_data %>% select(-FBndID)

# Save the training and validation sets
write_rds(train_data_rf, "data/rf_inputs/rf_train_data_0521.rds")
write_rds(val_data_rf, "data/rf_inputs/rf_val_data_0521.rds")


# --- 4. RANDOM FOREST MODEL TRAINING ---###########
 ------------------------------------------------------------------------------
# Train the Random Forest classifier on the prepared training data.

train_data_rf <- read_rds("data/rf_inputs/rf_train_data_0521.rds")

set.seed(123) # for reproducible model training
rf_model <- randomForest(
  obs ~ .,
  data = train_data_rf,
  ntree = 500,
  importance = TRUE
)

# Save the trained model object
write_rds(rf_model, "data/rf_model/rf_model_0521.rds")

rf_model<-read_rds("data/rf_model/rf_model_0521.rds")


# --- 5. MODEL EVALUATION ---
 ------------------------------------------------------------------------------
# Evaluate the model's performance on both training and validation sets.

rf_model <- read_rds("data/rf_model/rf_model_0521.rds")
val_data_rf <- read_rds("data/rf_inputs/rf_val_data_0521.rds")

# Evaluate on training data (often high due to overfitting, good for a baseline)
train_pred <- predict(rf_model, newdata = train_data_rf)
train_cm <- confusionMatrix(train_pred, train_data_rf$obs)
cat("--- Training Performance ---\n")
print(train_cm)

# Evaluate on validation data (a more realistic measure of performance)
val_pred <- predict(rf_model, newdata = val_data_rf)
val_cm <- confusionMatrix(val_pred, val_data_rf$obs)
cat("\n--- Validation Performance ---\n")
print(val_cm)



# --- 5. Spatial Mapping of Predicted Prevented Planting---############
## loading the acpf map
## loading the huc4 map 

huc8_sub_sf<-read_sf("gis/WBDHU8.shp") %>% 
  st_transform(5070) %>% 
  filter(Name != "Huron")  # Exclude rows where 'Name' equals 'Huron'.

## loading county map
county_huc4_sf<-read_sf("gis/county_huc4.shp")
county_huc4_sub_sf=st_intersection(county_huc4_sf, huc8_sub_sf) 

fbnd_sf<-st_read("gis/welb_merged_acpf.shp") %>%
   dplyr::select("FBndID","maj19") %>%
   mutate(obs = ifelse(maj19 == "I", "PP", "P")) %>%
   st_transform(5070)

# Read your spatial shapefile if not already loaded
val_predict=val_predict%>%
  mutate(val_pred_r=ifelse(val_pred=="I","PP","P"))

train_predict=train_predict%>%
  mutate(train_pred_r=ifelse(train_pred=="I","PP","P"))

# Join prediction results to spatial layer
fbnd_sf_pred <- fbnd_sf %>%
  left_join(val_predict, by = "FBndID") %>%
  left_join(train_predict, by = "FBndID")

fbnd_sf_pred <- fbnd_sf_pred %>%
  mutate(
    combined_pred = coalesce(train_pred_r, val_pred_r)  # use train_pred if not NA, otherwise val_pred
  )

fbnd_sf_pred$agreement <- ifelse(
  fbnd_sf_pred$obs == fbnd_sf_pred$combined_pred,
  "Correct",   # Prediction matches observation
  "Incorrect"  # Prediction does not match
)

agreement_colors <- c(
  "Correct" = "#2E8B57",   # green
  "Incorrect" = "#D73027"  # red
)


library(ggplot2)
# Common theme for maps
map_theme <- theme_minimal(base_size = 14) +
  theme(
    axis.text = element_blank(),
    axis.title = element_blank(),
    panel.grid = element_blank(),
    panel.border = element_blank(),
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    legend.title = element_text(size = 13, face = "bold"),
    legend.text = element_text(size = 12),
    legend.position = "bottom"
  )

# Consistent color palette (colorblind-friendly alternative)
# Color palette

pp_colors <- c(
  "PP" = "lightgray",   # medium gray
  "P"  = "green"    # dark green
)

boundary_colors <- c(
  "County boundary" = "red",
  "HUC8 boundary"   = "black"
)

library(ggplot2)
p_pred <- ggplot(fbnd_sf_pred) +
  geom_sf(aes(fill = obs), color = NA) +
  geom_sf(data = county_huc4_sub_sf, aes(color = "County boundary"), fill = NA, linewidth = 0.3) +
  geom_sf(data = huc8_sub_sf, aes(color = "HUC8 boundary"), fill = NA, linewidth = 0.5) +
  scale_fill_manual(values = pp_colors,
                    labels = c("PP" = "Prevented Planting", "P" = "Planting"),
                    name = "") +
  scale_color_manual(values = boundary_colors, name = NULL) +
  labs(title = "(a)") +
  map_theme


p_combined <- ggplot(fbnd_sf_pred) +
  geom_sf(aes(fill = combined_pred), color = NA) +
  geom_sf(data = county_huc4_sub_sf, aes(color = "County boundary"), fill = NA, linewidth = 0.3) +
  geom_sf(data = huc8_sub_sf, aes(color = "HUC8 boundary"), fill = NA, linewidth = 0.5) +
  scale_fill_manual(values = pp_colors,
                    labels = c("PP" = "Prevented Planting", "P" = "Planting"),
                    name = "") +
  scale_color_manual(values = boundary_colors, name = NULL) +
  labs(title = "(b) Predicted Planting Status (Random Forest)") +
  map_theme +
  theme(legend.position = "NULL")   # Place the legend at the bottom


agreement_colors <- c(
  "Correct" = "#2E8B57",   # green
  "Incorrect" = "#D73027"  # red
)


library(ggplot2)

p_agreement <- ggplot(fbnd_sf_pred) +
  geom_sf(aes(fill = agreement), color = NA) +   # Fields colored by accuracy
  geom_sf(data = county_huc4_sub_sf, aes(color = "County boundary"),
          fill = NA, linewidth = 0.3) +
  geom_sf(data = huc8_sub_sf, aes(color = "HUC8 boundary"),
          fill = NA, linewidth = 0.5) +
  scale_fill_manual(values = agreement_colors, name = "Prediction Accuracy") +
  scale_color_manual(values = boundary_colors, name = NULL) +
  labs(title = "(b)") +
  map_theme


# 1. Observed
library(patchwork)

combined_map <- (p_pred | p_agreement) +
  plot_layout(guides = "collect") +   # Collect legends once
  plot_annotation(
    title = "",
    theme = theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5))
  ) & theme(legend.position = "bottom")   # Place the legend at the bottom

#combined_map

ggsave("figures/fig3_rf_model.jpg", combined_map, width = 12, height = 6)


# --- 6. FEATURE IMPORTANCE ANALYSIS ---############
 ------------------------------------------------------------------------------
# Part A: Standard Gini-based Importance from the Random Forest model.

importance_df <- as.data.frame(randomForest::importance(rf_model)) %>%
  tibble::rownames_to_column("Variable")

# Define variable groups for coloring the plot
variable_groups <- list(
  Soil = c("Ksat", "clay", "nccpi_all"),
  Topographic = c("elev", "wslope", "wet", "acc_d8"),
  Crop = c("men_RCI", "Max_Prob", "First_Choice", "Second_Choice"),
  Tile = c("seetile"),
  Climate = c("pr_m3", "pr_m4", "pr_m5", "pr_m6", "deficit_m3", "deficit_m4", "deficit_m5", "deficit_m6"),
  `Census of Agriculture` = c("FERT_ACRES", "INCOME_PER_OP", "PCT_FARMING", "median_acres_op", "PCT_RES", "PCT_RENTED", "Farmer_ages", "MACH_ACRES", "CHEM_ACRES", "LAB_ACRES", "FED_ACRES", "PER_INSUR")
)

# Map variables to their groups
importance_df$Class <- "Other"
for (group_name in names(variable_groups)) {
  importance_df$Class[importance_df$Variable %in% variable_groups[[group_name]]] <- group_name
}

# Create a mapping for more readable variable names
rename_map <- c(
  men_RCI = "RCI", Max_Prob = "Prob. of First Crop", First_Choice = "First Crop Choice",
  Second_Choice = "Second Crop Choice", deficit_m3 = "Climate Moist. (Mar)", deficit_m4 = "Climate Moist. (Apr)",
  deficit_m5 = "Climate Moist.(May)", deficit_m6 = " Climate Moist.(Jun)", pr_m3 = "Precip (Mar)",
  pr_m4 = "Precip (Apr)", pr_m5 = "Precip (May)", pr_m6 = "Precip (Jun)",
  Acres = "Field Size", wslope = "Weighted Slope", elev = "Elevation", wet = "Wetness Index",
  acc_d8 = "Drainage Area", seetile = "Tile Drainage (SEETILE)", nccpi_all = "Crop Productivity Index",
  FERT_ACRES = "Fertilizer $/Acre", INCOME_PER_OP = "Income/Operation", PCT_FARMING = "% Farming Occupation",
  median_acres_op = "Median Farm Size", PCT_RES = "% Residential", PCT_RENTED = "% Rented Land",
  Farmer_ages = "Avg. Farmer Age", MACH_ACRES = "Machinery $/Acre", CHEM_ACRES = "Chemicals $/Acre",
  LAB_ACRES = "Labor $/Acre", FED_ACRES = "Federal Payments $/Acre", PER_INSUR = "% Insured Acres"
)

importance_df$rename_variable <- recode(importance_df$Variable, !!!rename_map, .default = importance_df$Variable)

# Plot Gini importance
p_gini <- ggplot(importance_df, aes(x = reorder(rename_variable, MeanDecreaseGini), y = MeanDecreaseGini, fill = Class)) +
  geom_col() +
  coord_flip() +
  labs(title = "Feature Importance (Gini)", x = "Variable", y = "Mean Decrease in Gini Impurity") +
  theme_minimal()
print(p_gini)

# --- Part B: SHAP (SHapley Additive exPlanations) Importance ---
# SHAP values provide more robust, locally interpretable feature importances.
# NOTE: This is computationally intensive. Parallel processing is recommended.

# Set up parallel backend
# Use the number of cores available, minus one for system stability
num_cores <- detectCores() - 1
registerDoParallel(cores = num_cores)

# Define the prediction wrapper function for fastshap
pred_wrapper <- function(object, newdata) {
  predict(object, newdata = newdata, type = "prob")[, "I"] # Probability of class 'I'
}

# Calculate SHAP values
set.seed(123)
shap_values <- fastshap::explain(
  rf_model,
  X = train_data_rf %>% select(-obs),
  newdata = val_data_rf %>% select(-obs),
  pred_wrapper = pred_wrapper,
  nsim = 100,
  parallel = TRUE
)

# Stop the parallel cluster
stopImplicitCluster()

# Calculate global feature importance from SHAP values (mean absolute SHAP)
shap_importance_df <- as.data.frame(colMeans(abs(shap_values)))
names(shap_importance_df) <- "Importance"
shap_importance_df <- shap_importance_df %>%
  tibble::rownames_to_column("Variable") %>%
  arrange(desc(Importance)) %>%
  left_join(importance_df %>% select(Variable, Class, rename_variable), by = "Variable") # Merge with group/rename info


shap_importance_df <- shap_importance_df %>%
  mutate(Class = case_when(
    Variable %in% c("Ksat", "clay", "nccpi_all") ~ "Soil",
    Variable %in% c("elev", "wslope", "wet", "acc_d8") ~ "Topographic",
    Variable %in% c("men_RCI","Max_Prob", "First_Choice", "Second_Choice") ~ "Crop",
    Variable %in% c("seetile") ~ "Tile",
    Variable %in% c("pr_m3", "pr_m4", "pr_m5", "pr_m6","deficit_m3","deficit_m4","deficit_m5","deficit_m6","soil_m3","soil_m4","soil_m5","soil_m6") ~ "Climate",
    Variable %in% c("FERT_ACRES","INCOME_PER_OP","PCT_FARMING","median_acres_op","PCT_RES","PCT_RENTED","Farmer_ages"    
                    ,"MACH_ACRES","CHEM_ACRES","LAB_ACRES","FED_ACRES","PER_INSUR")~ "Census of Agriculture",
    TRUE ~ "Other"
    
  ))

# Save SHAP importance results
write_csv(shap_importance_df, "data/shape_variable_importance_0521.csv")

shap_importance_df<-read_csv("data/shape_variable_importance_0521.csv")

#shap_importance_df$rename_variable <- recode(shap_importance_df$Variable, !!!rename_map, .default = shap_importance_df$Variable)

# Plot SHAP-based feature importance

shap_plot <- ggplot(shap_importance_df, 
                    aes(x = reorder(rename_variable, Importance),
                        y = Importance,
                        fill = Class)) +
  geom_col(width = 0.7) +
  coord_flip() +
  scale_fill_manual(values = c(
    "Soil"                = "#8B4513",  # saddle brown
    "Topographic"         = "#FFD700",  # gold
    "Crop"                = "#228B22",  # forest green
    "Tile"                = "#6A5ACD",  # slate blue
    "Climate"             = "#1E90FF",  # dodger blue
    "Census of Agriculture" = "#FFA07A",# light salmon
    "Other"               = "#B0B0B0"   # gray
  )) +
  labs(title = "Feature Importance Based on SHAP Values",
       x     = "Feature",
       y     = "Mean |SHAP value|",
       fill  = NULL) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title   = element_text(face = "bold", size = 16, hjust = 0.5),
    axis.text.y  = element_text(size = 12),
    axis.title   = element_text(size = 14),
    legend.position = "bottom"
  )

print(shap_plot)


ggsave("figures/fig4_shap_plot.jpg", plot = shap_plot, width =10, height = 10, dpi = 300)


# --- 7. ACCUMULATED LOCAL EFFECTS (ALE) PLOTS ---#########
 ------------------------------------------------------------------------------
# ALE plots show how a feature's value affects the model's prediction on average,
# after accounting for correlations with other features.



# Create a predictor object for the 'iml' package
predictor_rf <- Predictor$new(
  model = rf_model,
  data = train_data_rf %>% select(-obs),
  y = train_data_rf$obs,
  type = "prob"
)

features <- shap_importance_df[1:12,1]
feature_labels <- shap_importance_df[1:12,4]
# Generate labels "(a)", "(b)", … up to "(j)" for 10 features:

feature_vars <- features$Variable
labels <- paste0("(", letters[1:length(feature_vars)], ")")

# Now create a named vector mapping variable → label
title_labels <- setNames(labels, feature_vars)

# Select the top 12 features from the SHAP analysis to plot
top_12_features <- shap_importance_df %>% top_n(12, Importance)

# Generate ALE plots for each of the top features
ale_plots <- lapply(top_12_features$Variable, function(f) {
  
  ale_eff <- FeatureEffect$new(predictor_rf, feature = f, method = "ale")
  
  # Filter for the class of interest ('I' for Implemented)
  ale_data <- ale_eff$results %>% filter(.class == "I")
  
  # Get the pretty name for the x-axis label
  x_label_name <- top_12_features$rename_variable[top_12_features$Variable == f]
  
  # Create the plot
  p <- ggplot(ale_data, aes(x = .data[[f]], y = .value)) +
    geom_line(color = "midnightblue", size = 1) +
    geom_point() + 
    labs(
      title = paste0(title_labels[f]),  # Set title using feature label
      x = x_label_name,
      y =  NULL  # Remove the y-axis title from individual plots
    ) +
    theme_minimal(base_size = 14) +
     theme(
      plot.title = element_text(hjust = 0, size = 16, face = "bold"),
      axis.title = element_text(size = 14),
      axis.text = element_text(size = 12),
      panel.grid.major = element_line(color = "grey80"),
      panel.grid.minor = element_blank()
    )
  
  return(p)
})
# 
# # Combine all ALE plots into a single figure and save
# combined_ale_plot <- wrap_plots(ale_plots, ncol = 4) +
#   plot_annotation(
#     title = "Accumulated Local Effects on the Probability of Prevented Planting",
#     theme = theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5))
#   )


# Arrange the 12 individual plots into a grid
ale_grid <- wrap_plots(ale_plots, ncol = 4)

# Create an empty plot that will serve as the shared y-axis label
y_axis_label_plot <- ggplot() +
  labs(y = "ALE on Prevented Planting") +
  theme_void() +
  theme(
    # Make the y-axis title visible, bold, and sized appropriately
    axis.title.y = element_text(
      angle = 90,
      vjust = 0.5,
      size = 16,
      face = "bold"
    )
  )

# Combine the label plot and the grid.
# `plot_layout` controls the relative widths, making the label plot very narrow.
final_plot <- y_axis_label_plot + ale_grid +
  plot_layout(widths = c(1, 25)) + # Give label plot 1/26th of the total width
  plot_annotation(
    title = "",
    theme = theme(plot.title = element_text(size = 20, face = "bold", hjust = 0.5))
  )

ggsave("figures/fig5_ale_plot.jpg", final_plot, width = 12, height = 10, dpi = 500)


################################ END OF SCRIPT #################################
