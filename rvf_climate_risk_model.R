
# ============================================================
# RIFT VALLEY FEVER CLIMATE-INFORMED RISK MODELLING
# Simulation-based methodological demonstration
# Author: NANTE RACHEL WANGI
# Date: November 2026
# ============================================================

# 1. INITIAL SETUP
set.seed(2026)

dir.create("data", showWarnings = FALSE)
dir.create("figures", showWarnings = FALSE)
dir.create("results", showWarnings = FALSE)

# 2. GENERATE SYNTHETIC MONTHLY SURVEILLANCE DATA
dates <- seq(as.Date("2015-01-01"),
             by = "month", length.out = 120)

n <- length(dates)
month <- as.integer(format(dates, "%m"))
year <- as.integer(format(dates, "%Y"))

# Seasonal rainfall pattern, in millimetres
rainfall <- pmax(0, 95 +
                   65 * sin(2 * pi * (month - 2) / 12) +
                   rnorm(n, 0, 35))

# Simulated normalized vegetation index
ndvi <- pmin(0.9, pmax(0.1,
                       0.25 + 0.002 * rainfall + rnorm(n, 0, 0.07)))

# Simulated livestock infection reports
livestock_reports <- rpois(
  n, lambda = exp(-0.5 + 0.012 * rainfall))

# Predictors available from the preceding month
rainfall_lag1 <- c(NA, head(rainfall, -1))
ndvi_lag1 <- c(NA, head(ndvi, -1))
livestock_lag1 <- c(NA, head(livestock_reports, -1))

# Synthetic outcome-generating mechanism
# This is assumed for demonstration, not estimated
# from actual RVF surveillance records.
linear_predictor <- -3.5 +
  0.014 * rainfall_lag1 +
  2.0 * ndvi_lag1 +
  0.13 * livestock_lag1

risk_probability <- plogis(linear_predictor)
outbreak_risk <- rbinom(n, 1, risk_probability)

rvf <- data.frame(
  date = dates,
  year = year,
  month = month,
  rainfall_mm = round(rainfall, 2),
  ndvi = round(ndvi, 3),
  livestock_reports = livestock_reports,
  rainfall_lag1 = round(rainfall_lag1, 2),
  ndvi_lag1 = round(ndvi_lag1, 3),
  livestock_lag1 = livestock_lag1,
  outbreak_risk = outbreak_risk
)

rvf <- na.omit(rvf)
rownames(rvf) <- NULL

write.csv(rvf, "data/rvf_simulated_data.csv",
          row.names = FALSE)

cat("Observations:", nrow(rvf), "\n")
cat("Elevated-risk months:",
    sum(rvf$outbreak_risk), "\n")

# 3. DESCRIPTIVE ANALYSIS
print(summary(rvf))
print(table(rvf$outbreak_risk))

# 4. CHRONOLOGICAL TRAINING-TEST SPLIT
train <- subset(rvf, year <= 2021)
test <- subset(rvf, year >= 2022)

cat("Training observations:", nrow(train), "\n")
cat("Testing observations:", nrow(test), "\n")

# 5. FIT TWO LOGISTIC REGRESSION MODELS
# Baseline model: rainfall only
baseline <- glm(
  outbreak_risk ~ rainfall_lag1,
  data = train, family = binomial()
)

# Full model: three lagged indicators
full_model <- glm(
  outbreak_risk ~ rainfall_lag1 +
    ndvi_lag1 + livestock_lag1,
  data = train, family = binomial()
)

print(summary(full_model))

# Save model coefficients as odds ratios
coef_table <- summary(full_model)$coefficients

odds_ratios <- data.frame(
  variable = rownames(coef_table),
  odds_ratio = exp(coef(full_model)),
  lower_95_ci = exp(coef(full_model) -
                      1.96 * coef_table[, 2]),
  upper_95_ci = exp(coef(full_model) +
                      1.96 * coef_table[, 2]),
  p_value = coef_table[, 4]
)

write.csv(odds_ratios,
          "results/model_odds_ratios.csv",
          row.names = FALSE)

# 6. PREDICT FUTURE TEST PERIOD
test$baseline_prob <- predict(
  baseline, newdata = test, type = "response")

test$full_prob <- predict(
  full_model, newdata = test, type = "response")

# 7. MODEL EVALUATION FUNCTIONS
evaluate <- function(actual, probability,
                     threshold = 0.5) {
  predicted <- as.integer(probability >= threshold)
  
  tp <- sum(actual == 1 & predicted == 1)
  tn <- sum(actual == 0 & predicted == 0)
  fp <- sum(actual == 0 & predicted == 1)
  fn <- sum(actual == 1 & predicted == 0)
  
  safe_divide <- function(a, b) {
    if (b == 0) return(NA_real_)
    a / b
  }
  
  c(
    sensitivity = safe_divide(tp, tp + fn),
    specificity = safe_divide(tn, tn + fp),
    accuracy = safe_divide(tp + tn, length(actual)),
    brier_score = mean((probability - actual)^2)
  )
}

# Calculate ROC AUC without additional packages
calculate_auc <- function(actual, probability) {
  positive <- probability[actual == 1]
  negative <- probability[actual == 0]
  
  if (length(positive) == 0 ||
      length(negative) == 0) return(NA_real_)
  
  mean(outer(positive, negative, ">")) +
    0.5 * mean(outer(positive, negative, "=="))
}

baseline_metrics <- evaluate(
  test$outbreak_risk, test$baseline_prob)

full_metrics <- evaluate(
  test$outbreak_risk, test$full_prob)

comparison <- data.frame(
  model = c("Rainfall baseline", "Full model"),
  rbind(baseline_metrics, full_metrics),
  auc = c(
    calculate_auc(test$outbreak_risk,
                  test$baseline_prob),
    calculate_auc(test$outbreak_risk,
                  test$full_prob)
  ),
  row.names = NULL
)

print(comparison)

write.csv(comparison,
          "results/model_comparison.csv",
          row.names = FALSE)

write.csv(test,
          "results/test_predictions.csv",
          row.names = FALSE)

# 8. FIGURE 1: RAINFALL OVER TIME
png("figures/rainfall_trend.png",
    width = 1100, height = 650)

plot(rvf$date, rvf$rainfall_mm,
     type = "l", lwd = 2,
     xlab = "Year", ylab = "Rainfall (mm)",
     main = "Simulated Monthly Rainfall")
dev.off()

# 9. FIGURE 2: OBSERVED SYNTHETIC RISK
png("figures/rvf_risk_over_time.png",
    width = 1100, height = 650)

plot(rvf$date, rvf$outbreak_risk,
     type = "h", lwd = 3,
     xlab = "Year",
     ylab = "Elevated RVF risk (0/1)",
     main = "Simulated RVF Risk Classification")
dev.off()

# 10. FIGURE 3: PREDICTED PROBABILITIES
png("figures/model_predictions.png",
    width = 1100, height = 650)

plot(test$date, test$full_prob,
     type = "l", lwd = 2, ylim = c(0, 1),
     xlab = "Date",
     ylab = "Predicted probability",
     main = "Out-of-Sample RVF Risk Predictions")

lines(test$date, test$baseline_prob,
      lty = 2, lwd = 2)

legend("topright",
       legend = c("Full model", "Rainfall baseline"),
       lty = c(1, 2), lwd = 2)
dev.off()

# 11. SAVE MODELS AND REPRODUCIBILITY DETAILS
saveRDS(full_model, "results/full_model.rds")
saveRDS(baseline, "results/baseline_model.rds")

capture.output(
  sessionInfo(),
  file = "results/session_info.txt"
)

cat("\nPROJECT ANALYSIS COMPLETED.\n")
cat("Check data/, figures/, and results/ folders.\n")
