################################################################################
#
#       K-MEANS CLUSTER ANALYSIS OF PREVENTED PLANTING DATA
#
# This script performs a cluster analysis on agricultural field data. It uses
# the top variables identified by a SHAP analysis, applies PCA for dimensionality
# reduction, and then runs k-means clustering to identify distinct field archetypes.
# Finally, it visualizes the characteristics of each cluster.
#
# Assumes the script is run from the root of an RStudio project where the
# 'data/' and 'figures/' directories exist.
#
################################################################################

# --- 1. SETUP: LOAD LIBRARIES ---
# ------------------------------------------------------------------------------
# Load all necessary packages for the analysis.
# For data manipulation and piping
library(dplyr)
library(tidyr)

# For reading CSV files
library(readr)

# For clustering and cluster validation/visualization
library(cluster)
library(factoextra)

# For creating dummy variables (one-hot encoding)
library(fastDummies)
# For creating and arranging complex plots
library(ggplot2)
library(patchwork)
library(tidytext) # For the reorder_within function

# --- 2. DATA LOADING AND PREPROCESSING ---
# ------------------------------------------------------------------------------
# Load the datasets, select features, and prepare them for analysis.

# Load the primary dataset and the SHAP variable importance list
data <- read_csv("data/rf_inputs/rf_inputs_0521.csv")
shap_importance_df <- read_csv("data/shape_variable_importance_0521.csv")

# Convert specified character columns to factors for modeling
data <- data %>%
  mutate(
    First_Choice = as.factor(First_Choice),
    Second_Choice = as.factor(Second_Choice),
    obs = factor(obs, levels = c("P", "I")) # 'P' for Planting, 'I' for prevented planting
  )

# Extract the names of the top 12 most important variables from the SHAP results
top_12_features <- shap_importance_df$Variable[1:12]

# Create the primary analysis dataframe by selecting key identifiers and top 12 features.
# Remove any rows with missing data to ensure PCA and k-means run without errors.
farm_data <- data %>%
  select(FBndID, obs, all_of(top_12_features)) %>%
  drop_na()

# Separate data into numeric and categorical types for differential treatment
numeric_data <- farm_data %>% select(where(is.numeric))

# One-hot encode the 'First_Choice' categorical variable. This converts the factor
# into a set of binary (0/1) columns, suitable for use in distance-based algorithms.
factor_data <- dummy_cols(
  .data = data.frame(First_Choice = farm_data$First_Choice),
  remove_first_dummy = TRUE,       # Avoid multicollinearity
  remove_selected_columns = TRUE   # Drop the original 'First_Choice' column
)


# --- 3. DIMENSIONALITY REDUCTION WITH PCA ---
# ------------------------------------------------------------------------------
# Apply Principal Component Analysis (PCA) to the numeric variables to handle
# multicollinearity and reduce the feature space.

# Perform PCA on the numeric data. 'scale. = TRUE' is crucial as it standardizes
# variables to have a mean of 0 and a standard deviation of 1 before analysis.
pca_result <- prcomp(numeric_data, scale. = TRUE)

# Inspect the summary to see the proportion of variance explained by each PC
summary(pca_result)

# --- Visualize PCA Results: Scree Plots ---
# These plots help decide how many Principal Components (PCs) to retain.
p1_scree <- fviz_eig(pca_result, addlabels = TRUE, ylim = c(0, 40), main = "Explained Variance (%)")
p2_eigen <- fviz_eig(pca_result, choice = "eigenvalue", addlabels = TRUE, ylim = c(0, 5), main = "Eigenvalues")

# Combine the two plots and save
combined_pca_plot <- (p1_scree | p2_eigen) +
  plot_annotation(
    title = "Scree Plots for PCA",
    tag_levels = 'a', tag_prefix = "(", tag_suffix = ")"
  )

combined_pca_plot
ggsave("figures/figS11_pca_variance.jpg", combined_pca_plot, width = 10, height = 5)


# --- Visualize PCA Results: Variable Loadings ---
# Loadings show how much each original variable contributes to a given PC.
loadings_df <- as.data.frame(pca_result$rotation)
loadings_df$variable <- rownames(loadings_df)

# Create a mapping for more readable variable names for the plot
rename_map <- c(
  men_RCI = "RCI", Max_Prob = "Prob. of First Crop", deficit_m3 = "Climate Moist. (Mar)",
  deficit_m4 = "Climate Moist.(Apr)", deficit_m5 = "Climate Moist.(May)", pr_m3 = "Precip (Mar)",
  pr_m4 = "Precip (Apr)", pr_m6 = "Precip (Jun)", elev = "Elevation",
  PER_INSUR = "% Insured Acres", median_acres_op = "Median Farm Size"
)

# Prepare the data for plotting: pivot to a long format and rename variables
loadings_long <- loadings_df %>%
  select(variable, PC1, PC2, PC3, PC4, PC5) %>%
  pivot_longer(cols = starts_with("PC"), names_to = "PC", values_to = "loading") %>%
  mutate(
    rename_variable = recode(variable, !!!rename_map),
    abs_loading = abs(loading)
  ) %>%
  # Use reorder_within to correctly order variables within each facet
  group_by(PC) %>%
  mutate(rename_variable = reorder_within(rename_variable, abs_loading, PC))

# Create the plot of variable contributions to the first 5 PCs
loadings_plot <- ggplot(loadings_long, aes(x = rename_variable, y = abs_loading, fill = PC)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  facet_wrap(~ PC, scales = "free_y", labeller = as_labeller(c(
    PC1 = "(a)",
    PC2 = "(b)",
    PC3 = "(c)",
    PC4 = "(d)",
    PC5 = "(e)"
  )))+
  scale_x_reordered() +
  labs(
    title = "Variable Contributions to the First Five Principal Components",
    x = "Variable",
    y = "Absolute Loading (|Contribution|)"
  ) +
  theme_minimal(base_size = 12) +
  theme(strip.text = element_text(face = "bold"))

loadings_plot
ggsave("figures/figS12_top5_pca.jpg", loadings_plot, width = 12, height = 7)


# --- 4. K-MEANS CLUSTERING ---
# ------------------------------------------------------------------------------
# Perform k-means clustering on the PCA scores and the encoded categorical data.

# Select the first 5 PCs, which capture a significant amount of variance.
pca_scores <- as.data.frame(pca_result$x[, 1:5])

# Combine the PCA scores (representing numeric data) with the one-hot encoded data.
cluster_data <- cbind(pca_scores, factor_data)

# OPTIONAL: Determine the optimal number of clusters. These functions can be
# computationally intensive. We will proceed with k=8 based on prior analysis.
#
# set.seed(123)
# farm_sample <- as_tibble(cluster_data) %>% sample_n(min(5000, nrow(cluster_data)))
# fviz_nbclust(farm_sample, kmeans, method = "wss")        # Elbow method
# fviz_nbclust(farm_sample, kmeans, method = "silhouette") # Silhouette method

# Run the k-means algorithm with k=8.
# 'nstart = 25' runs the algorithm 25 times with different random starting
# points and selects the best outcome, improving the stability of the result.
set.seed(123) # for reproducible clustering
km_result <- kmeans(cluster_data, centers = 8, nstart = 25)

# For final analysis, combine the original data with cluster assignments and PCs.
cluster_data_full <- farm_data %>%
  mutate(
    cluster = factor(km_result$cluster),
    PC1 = pca_scores$PC1,
    PC2 = pca_scores$PC2,
    PC3 = pca_scores$PC3,
    PC4 = pca_scores$PC4,
    PC5 = pca_scores$PC5
  )

# Save the complete clustered dataset for future use (e.g., mapping, further analysis).
write_csv(cluster_data_full, file = "data/cluster_full_data_0521.csv")


cluster_data_full<-read_csv(file = "data/cluster/cluster_full_data_0521.csv")

# --- 5. VISUALIZATION AND INTERPRETATION OF CLUSTERS ---
# ------------------------------------------------------------------------------
# Create plots to understand the characteristics of each cluster.

# Define a consistent color palette for the clusters
cluster_colors <- c(
  "1" = "#1b9e77", "2" = "#d95f02", "3" = "#7570b3", "4" = "#e7298a",
  "5" = "#66a61e", "6" = "#e6ab02", "7" = "#a6761d", "8" = "#666666"
)

# --- Plot 1: Boxplots of Variable Distributions per Cluster ---

# First, calculate the percentage of "Implemented" (obs == 'I') fields in each cluster.
# This will be used to label the plots.
status_summary <- cluster_data_full %>%
  group_by(cluster, obs) %>%
  summarise(count = n(), .groups = 'drop_last') %>%
  mutate(percent = count / sum(count) * 100) %>%
  ungroup()

# Filter for the 'I' percentage and format for labeling
labels_df <- status_summary %>%
  filter(obs == "I") %>%
  select(cluster, percent) %>%
  mutate(label = paste0(round(percent, 0), "% PP"))


# Pivot the data into a long format for facetted plotting with ggplot
farm_long_sub <- cluster_data_full %>%
  select(
    "cluster", "men_RCI", "elev","pr_m3", "PER_INSUR", "median_acres_op", "pr_m6"
  ) %>%
  rename_with(~ recode(., !!!rename_map), .cols = any_of(names(rename_map))) %>%
  pivot_longer(
    cols = -cluster,
    names_to = "numeric_variable",
    values_to = "value"
  )


label_positions <- farm_long_sub  %>%
  group_by(cluster, numeric_variable) %>%
  summarise(y_pos = max(value, na.rm = TRUE), .groups = "drop") %>%
  left_join(
    status_summary %>%
      select(cluster, obs, percent) %>%
      mutate(percent = round(percent, 1)) %>%
      tidyr::pivot_wider(names_from = obs, values_from = percent) %>%
      mutate(label = paste0(I, "%")),
    by = "cluster"
  )
# Create the boxplot visualization
p_boxplot <- ggplot(farm_long_sub, aes(x = as.factor(cluster), y = value, fill = as.factor(cluster))) +
  geom_boxplot(outlier.shape = NA) +
  facet_wrap(~ numeric_variable, scales = "free_y", ncol = 3) +
  # Add the percentage labels to the top of each plot facet
  geom_text(
    data = label_positions,
    aes(x = as.factor(cluster), y = y_pos * 1.05, label = label),
    inherit.aes = FALSE, size = 2.5, vjust = 1.5
  ) +
  scale_fill_manual(values = cluster_colors, guide = "none") +
  labs(
    title = "Distribution of Key Variables Across Clusters",
    subtitle = "",
    x = "Cluster",
    y = "Value"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    strip.text = element_text(face = "bold", size = 10),
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title.position = "plot"
  )

# Print and save the plot
print(p_boxplot)

ggsave("figures/fig6_cluster_variables_distributions.jpg", plot = p_boxplot, width = 12, height = 8, dpi = 300)

# --- Plot 2: Spatial Map of Clusters (Requires External Data) ---

# NOTE: The following code for plotting a map is provided for context but is
# commented out. To run it, you must first load your spatial data (e.g.,
# shapefiles for fields, counties, HUCs) and join them with the
# 'cluster_data_full' dataframe using a common identifier like 'FBndID'.
#
library(sf)

## loading the huc4 map 
huc8_sub_sf<-read_sf("gis/WBDHU8.shp") %>% 
  st_transform(5070) %>% 
  filter(Name != "Huron")  # Exclude rows where 'Name' equals 'Huron'.

## loading county map
## loading the huc4 map 
county_huc4_sf<-read_sf("gis/county_huc4.shp")
county_huc4_sub_sf=st_intersection(county_huc4_sf, huc8_sub_sf) 

#write_sf(county_huc4_sub_sf,"gis/county_huc4_sub_sf.shp")

fbnd_sf <- st_read("gis/welb_merged_acpf.shp") %>%
  dplyr::select("FBndID", "HUC8", "HUC10", "HUC12") %>%
  st_transform(5070)

## merging data
farm_cluster<-read_csv("data/cluster/cluster_full_data_0521.csv")

cluster_sf<-merge(fbnd_sf,farm_cluster,by="FBndID")

p_map <- ggplot() +
  
  # Cluster fill
  geom_sf(data = cluster_sf,
          aes(fill = factor(cluster)),
          color = NA) +
  
  # County boundary (red)
  geom_sf(data = county_huc4_sub_sf,
          aes(color = "County boundary"),
          fill = NA,
          linewidth = 0.3) +
  
  # HUC8 boundary (black)
  geom_sf(data = huc8_sub_sf,
          aes(color = "HUC8 boundary"),
          fill = NA,
          linewidth = 0.5) +
  
  scale_fill_manual(values = cluster_colors,
                    name = "Cluster") +
  
  scale_color_manual(
    name = "Boundary",
    values = c(
      "County boundary" = "red",
      "HUC8 boundary"   = "black"
    )
  ) +
  
  labs(
    title = "Clusters Based on SHAP Importance Analysis",
    subtitle = "Spatial distribution of clustered regions"
  ) +
  
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "right",
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    panel.grid = element_blank()
  )

# print(p_map)
ggsave("figures/fig6_cluster_shap.jpg", plot =p_map, width = 8, height = 6, dpi = 400)


## including all variables ##################


# Pivot the data into a long format for facetted plotting with ggplot
farm_cluster_sub<-cluster_data_full %>%
  select("men_RCI","Max_Prob","First_Choice","elev","pr_m3",
         "deficit_m5","deficit_m3",  "deficit_m4",   "pr_m4","PER_INSUR","median_acres_op", "pr_m6",
         "cluster")

farm_long<- farm_cluster_sub %>%
   rename_with(~ recode(., !!!rename_map), .cols = any_of(names(rename_map))) %>%
  pivot_longer(
    cols = -c(cluster,`First_Choice`),  # or: -c(cluster, First_Choice)
    names_to = "numeric_variable",
    values_to = "value"
  )

label_positions <- farm_long  %>%
  group_by(cluster, numeric_variable) %>%
  summarise(y_pos = max(value, na.rm = TRUE), .groups = "drop") %>%
  left_join(
    status_summary %>%
      select(cluster, obs, percent) %>%
      mutate(percent = round(percent, 1)) %>%
      tidyr::pivot_wider(names_from = obs, values_from = percent) %>%
      mutate(label = paste0(I, "%")),
    by = "cluster"
  )

# Create the boxplot visualization
p_boxplot_full <- ggplot(farm_long, aes(x = as.factor(cluster), y = value, fill = as.factor(cluster))) +
  geom_boxplot(outlier.shape = NA) +
  facet_wrap(~ numeric_variable, scales = "free_y", ncol = 3) +
  # Add the percentage labels to the top of each plot facet
  geom_text(
    data = label_positions,
    aes(x = as.factor(cluster), y = y_pos * 1.1, label = label),
    inherit.aes = FALSE, size = 2.5, vjust = 1.5
  ) +
  scale_fill_manual(values = cluster_colors, guide = "none") +
  labs(
    title = "",
    subtitle = "",
    x = "Cluster",
    y = "Value"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    strip.text = element_text(face = "bold", size = 10),
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title.position = "plot"
  )

# Print and save the plot
print(p_boxplot_full)

ggsave("figures/figS13_variable_cluster.jpg", plot = p_boxplot_full, width = 12, height = 8, dpi = 300)



################################ END OF SCRIPT #################################