# ============================================================================
# PROJECT 4: CUSTOMER SUPPORT TICKET CLASSIFIER & PRIORITIZER
# Complete Text Mining & Classification Workflow in R - FINAL VERSION
# ============================================================================

# Install required packages (run once)
install_packages <- function() {
  packages <- c(
    "readtext", "rvest", "httr", "xml2", "jsonlite",
    "dplyr", "magrittr", "stringr", "tidytext", 
    "tm", "SnowballC", "textstem", "tokenizers",
    "tidyr", "caret", "randomForest", "e1071",
    "pROC", "ggplot2", "wordcloud", "RColorBrewer",
    "lubridate", "tibble"
  )
  
  new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
  if(length(new_packages)) install.packages(new_packages)
}

# Load libraries
library(readtext)
library(rvest)
library(httr)
library(xml2)
library(jsonlite)
library(dplyr)
library(magrittr)
library(stringr)
library(tidytext)
library(tm)
library(SnowballC)
library(textstem)
library(tokenizers)
library(tidyr)
library(caret)
library(randomForest)
library(e1071)
library(pROC)
library(ggplot2)
library(wordcloud)
library(lubridate)
library(tibble)

# ============================================================================
# 1. DATA COLLECTION - REAL WEB SCRAPING
# ============================================================================

# Function 1: Scrape Reddit tech support posts (public data)
scrape_reddit_support <- function(subreddit = "techsupport", limit = 100) {
  cat("Scraping Reddit r/", subreddit, "...\n", sep = "")
  
  tryCatch({
    # Reddit's JSON feed (public, no auth needed)
    url <- paste0("https://www.reddit.com/r/", subreddit, "/new.json?limit=", limit)
    
    response <- GET(url, 
                    user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"))
    
    if (status_code(response) == 200) {
      data <- fromJSON(rawToChar(response$content))
      posts <- data$data$children$data
      
      df <- tibble(
        ticket_id = paste0("RED_", seq_len(nrow(posts))),
        customer_id = paste0("CUST_", substr(posts$author, 1, 8)),
        title = posts$title,
        description = posts$selftext,
        source = "Reddit",
        timestamp = as.POSIXct(posts$created_utc, origin = "1970-01-01"),
        score = posts$score
      )
      
      cat("✓ Scraped", nrow(df), "Reddit posts\n")
      return(df)
    }
  }, error = function(e) {
    cat("Error scraping Reddit:", e$message, "\n")
    return(NULL)
  })
}

# Function 2: Scrape StackOverflow questions
scrape_stackoverflow <- function(tag = "python", limit = 100) {
  cat("Scraping StackOverflow [", tag, "] questions...\n", sep = "")
  
  tryCatch({
    url <- paste0("https://api.stackexchange.com/2.3/questions?order=desc&sort=activity&tagged=",
                  tag, "&site=stackoverflow&pagesize=", limit, "&filter=withbody")
    
    response <- GET(url, user_agent("Mozilla/5.0"))
    
    if (status_code(response) == 200) {
      data <- fromJSON(rawToChar(response$content))
      questions <- data$items
      
      df <- tibble(
        ticket_id = paste0("SO_", questions$question_id),
        customer_id = paste0("USER_", questions$owner$user_id),
        title = questions$title,
        description = questions$body,
        source = "StackOverflow",
        timestamp = as.POSIXct(questions$creation_date, origin = "1970-01-01"),
        score = questions$score
      )
      
      cat("✓ Scraped", nrow(df), "StackOverflow questions\n")
      return(df)
    }
  }, error = function(e) {
    cat("Error scraping StackOverflow:", e$message, "\n")
    return(NULL)
  })
}

# Function 3: Scrape Hacker News posts
scrape_hackernews <- function(limit = 100) {
  cat("Scraping Hacker News...\n")
  
  tryCatch({
    # Get top story IDs
    url_top <- "https://hacker-news.firebaseio.com/v0/topstories.json"
    response <- GET(url_top)
    story_ids <- fromJSON(rawToChar(response$content))[1:limit]
    
    stories <- list()
    for (id in story_ids) {
      url_story <- paste0("https://hacker-news.firebaseio.com/v0/item/", id, ".json")
      story_response <- GET(url_story)
      if (status_code(story_response) == 200) {
        stories[[length(stories) + 1]] <- fromJSON(rawToChar(story_response$content))
      }
      Sys.sleep(0.1) # Rate limiting
    }
    
    df <- tibble(
      ticket_id = paste0("HN_", sapply(stories, function(x) x$id)),
      customer_id = paste0("USER_", sapply(stories, function(x) ifelse(is.null(x$by), "anonymous", x$by))),
      title = sapply(stories, function(x) ifelse(is.null(x$title), "", x$title)),
      description = sapply(stories, function(x) ifelse(is.null(x$text), x$title, x$text)),
      source = "HackerNews",
      timestamp = as.POSIXct(sapply(stories, function(x) x$time), origin = "1970-01-01"),
      score = sapply(stories, function(x) ifelse(is.null(x$score), 0, x$score))
    )
    
    cat("✓ Scraped", nrow(df), "Hacker News posts\n")
    return(df)
  }, error = function(e) {
    cat("Error scraping Hacker News:", e$message, "\n")
    return(NULL)
  })
}

# Collect all data
cat("\n=== STARTING DATA COLLECTION ===\n\n")
reddit_data <- scrape_reddit_support("techsupport", 100)
stackoverflow_data <- scrape_stackoverflow("python", 100)
hackernews_data <- scrape_hackernews(100)

# Combine all datasets
all_tickets <- bind_rows(
  reddit_data,
  stackoverflow_data,
  hackernews_data
) %>%
  filter(!is.na(description), description != "", nchar(description) > 20)

cat("\n✓ Total tickets collected:", nrow(all_tickets), "\n")

# ============================================================================
# 2. TEXT DATA READING WITH READTEXT (if you have local files)
# ============================================================================

read_local_tickets <- function(directory) {
  if (dir.exists(directory)) {
    tickets <- readtext(paste0(directory, "/*.txt"))
    cat("Read", nrow(tickets), "local ticket files\n")
    return(tickets)
  }
  return(NULL)
}

# ============================================================================
# 3. TEXT PRE-PROCESSING & BASIC MANIPULATION
# ============================================================================

cat("\n=== STARTING TEXT PRE-PROCESSING ===\n\n")

# Function to clean and standardize text using PIPES
preprocess_pipeline <- function(text) {
  text %>%
    # Remove HTML tags (from web scraping)
    str_replace_all("<[^>]+>", " ") %>%
    # Remove URLs
    str_replace_all("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", " ") %>%
    # Remove email addresses
    str_replace_all("\\S+@\\S+", " ") %>%
    # Remove HTML entities like &quot; &amp; etc
    str_replace_all("&[a-z]+;", " ") %>%
    # Convert to lowercase
    str_to_lower() %>%
    # Remove special characters but keep spaces
    str_replace_all("[^a-z0-9\\s]", " ") %>%
    # Remove extra whitespace
    str_replace_all("\\s+", " ") %>%
    str_trim()
}

# Apply preprocessing pipeline
all_tickets <- all_tickets %>%
  mutate(
    clean_title = preprocess_pipeline(title),
    clean_description = preprocess_pipeline(description),
    combined_text = paste(clean_title, clean_description, sep = " ")
  )

cat("✓ Text preprocessing completed\n")

# ============================================================================
# 4. REGULAR EXPRESSIONS - Extract Patterns
# ============================================================================

cat("\n=== EXTRACTING PATTERNS WITH REGEX ===\n\n")

# Extract error codes (e.g., ERROR_404, ERR-500)
all_tickets$error_codes <- str_extract_all(
  all_tickets$description,
  "(?i)(error[_\\s-]?\\d{3,4}|err[_\\s-]?\\d{3,4})"
) %>% sapply(function(x) paste(x, collapse = ", "))

# Extract product names (common tech products)
product_patterns <- c("windows", "mac", "linux", "android", "ios", 
                      "chrome", "firefox", "safari", "edge",
                      "python", "java", "javascript", "sql")
all_tickets$products <- sapply(all_tickets$combined_text, function(text) {
  found <- product_patterns[sapply(product_patterns, function(p) str_detect(text, p))]
  paste(found, collapse = ", ")
})

# Extract version numbers (e.g., v1.2.3, version 2.0)
all_tickets$versions <- str_extract(
  all_tickets$description,
  "(?i)(v|version)?\\s?\\d+\\.\\d+(\\.\\d+)?"
)

cat("✓ Pattern extraction completed\n")
cat("  - Error codes extracted from", sum(all_tickets$error_codes != ""), "tickets\n")
cat("  - Products identified in", sum(all_tickets$products != ""), "tickets\n")

# ============================================================================
# 5. ADVANCED TEXT PRE-PROCESSING TECHNIQUES
# ============================================================================

cat("\n=== TOKENIZATION & N-GRAMS ===\n\n")

# Tokenization
tokens <- all_tickets %>%
  select(ticket_id, combined_text) %>%
  unnest_tokens(word, combined_text)

cat("✓ Created", nrow(tokens), "tokens\n")

# Remove stopwords
data(stop_words)
tokens_clean <- tokens %>%
  anti_join(stop_words, by = "word") %>%
  filter(str_length(word) > 2) %>%
  # Filter out common noise words
  filter(!word %in% c("quot", "amp", "lt", "gt", "nbsp"))

cat("✓ Removed stopwords, remaining:", nrow(tokens_clean), "tokens\n")

# Stemming
tokens_clean$word_stem <- wordStem(tokens_clean$word, language = "english")

# Lemmatization
tokens_clean$word_lemma <- lemmatize_words(tokens_clean$word)

cat("✓ Stemming and lemmatization completed\n")

# Create bigrams (2-grams) for common issue phrases
bigrams <- all_tickets %>%
  select(ticket_id, combined_text) %>%
  unnest_tokens(bigram, combined_text, token = "ngrams", n = 2) %>%
  filter(!is.na(bigram)) %>%
  # Filter out bigrams containing noise words
  filter(!str_detect(bigram, "quot|amp|lt|gt|nbsp"))

# Find most common bigrams
top_bigrams <- bigrams %>%
  count(bigram, sort = TRUE) %>%
  head(20)

cat("\n✓ Top 10 bigrams (common issue phrases):\n")
print(top_bigrams %>% head(10))

# Create trigrams (3-grams)
trigrams <- all_tickets %>%
  select(ticket_id, combined_text) %>%
  unnest_tokens(trigram, combined_text, token = "ngrams", n = 3) %>%
  filter(!is.na(trigram)) %>%
  # Filter out trigrams containing noise words
  filter(!str_detect(trigram, "quot|amp|lt|gt|nbsp")) %>%
  count(trigram, sort = TRUE) %>%
  head(10)

cat("\n✓ Top 5 trigrams:\n")
print(trigrams %>% head(5))

# ============================================================================
# 6. BAG OF WORDS & CORPUS ANALYTICS
# ============================================================================

cat("\n=== CREATING BAG OF WORDS MODEL ===\n\n")

# Create corpus
corpus <- Corpus(VectorSource(all_tickets$combined_text))

# Custom stopwords including HTML entities
custom_stopwords <- c(stopwords("english"), "quot", "amp", "lt", "gt", "nbsp", "x2f", "x27")

# Create Document-Term Matrix (BoW)
dtm <- DocumentTermMatrix(corpus, control = list(
  tolower = TRUE,
  removePunctuation = TRUE,
  removeNumbers = FALSE,
  stopwords = custom_stopwords,
  stemming = TRUE,
  wordLengths = c(3, 15),
  bounds = list(global = c(3, Inf)) # Words appearing in at least 3 documents
))

cat("✓ Document-Term Matrix created\n")
cat("  - Documents:", dtm$nrow, "\n")
cat("  - Terms:", dtm$ncol, "\n")

# Calculate actual sparsity
sparsity <- 100 * (1 - (length(dtm$i) / (dtm$nrow * dtm$ncol)))
cat("  - Sparsity:", round(sparsity, 2), "%\n")

# Remove sparse terms
dtm_reduced <- removeSparseTerms(dtm, 0.99)
cat("✓ Reduced DTM terms:", dtm_reduced$ncol, "\n")

# Convert to matrix for modeling
bow_matrix <- as.matrix(dtm_reduced)
bow_df <- as.data.frame(bow_matrix)

# Make valid column names for Random Forest
colnames(bow_df) <- make.names(colnames(bow_df))

# ============================================================================
# 7. CORPUS ANALYTICS
# ============================================================================

cat("\n=== CORPUS ANALYTICS ===\n\n")

# Most frequent terms
term_freq <- colSums(bow_matrix)
term_freq_sorted <- sort(term_freq, decreasing = TRUE)

cat("✓ Top 15 most frequent terms:\n")
print(head(term_freq_sorted, 15))

# Term frequency by source
source_terms <- all_tickets %>%
  select(source, combined_text) %>%
  unnest_tokens(word, combined_text) %>%
  anti_join(stop_words, by = "word") %>%
  filter(!word %in% c("quot", "amp", "lt", "gt", "nbsp", "x2f", "x27")) %>%
  count(source, word, sort = TRUE)

cat("\n✓ Top terms by source:\n")
print(source_terms %>% group_by(source) %>% slice_max(n, n = 3))

# Time-based analysis
all_tickets$hour <- hour(all_tickets$timestamp)
all_tickets$day_of_week <- wday(all_tickets$timestamp, label = TRUE)

cat("\n✓ Tickets by day of week:\n")
print(table(all_tickets$day_of_week))

# ============================================================================
# 8. MANUAL LABELING FOR CLASSIFICATION
# ============================================================================

cat("\n=== CREATING LABELS FOR CLASSIFICATION ===\n\n")

# Create labels based on keywords (simulating manual labeling)
classify_ticket <- function(text) {
  text_lower <- tolower(text)
  
  # Bug keywords
  if (str_detect(text_lower, "bug|error|crash|broken|not working|issue|problem|fail")) {
    return("bug")
  }
  # Feature request keywords
  else if (str_detect(text_lower, "feature|request|suggestion|add|implement|would like|could you|enhancement")) {
    return("feature_request")
  }
  # Complaint keywords
  else if (str_detect(text_lower, "complaint|slow|terrible|worst|bad|disappointed|frustrated|angry")) {
    return("complaint")
  }
  # Query keywords
  else if (str_detect(text_lower, "how to|how do|question|help|what is|why|when|where|can someone")) {
    return("query")
  }
  else {
    return("query") # Default category
  }
}

all_tickets$category <- sapply(all_tickets$combined_text, classify_ticket)

cat("✓ Category distribution:\n")
print(table(all_tickets$category))

# ============================================================================
# 9. PREPARE DATA FOR MODELING
# ============================================================================

cat("\n=== PREPARING MODELING DATASET ===\n\n")

# Combine BoW features with labels
modeling_data <- bow_df
modeling_data$category <- factor(all_tickets$category[1:nrow(bow_df)])

# Remove rows with NA
modeling_data <- modeling_data[complete.cases(modeling_data), ]

# Handle imbalanced classes - merge complaint into bug
cat("✓ Original category counts:\n")
print(table(modeling_data$category))

modeling_data$category <- as.character(modeling_data$category)
modeling_data$category[modeling_data$category == "complaint"] <- "bug"
modeling_data$category <- factor(modeling_data$category)

cat("\n✓ Adjusted category counts (complaint merged into bug):\n")
print(table(modeling_data$category))

cat("\n✓ Modeling dataset prepared\n")
cat("  - Features:", ncol(modeling_data) - 1, "\n")
cat("  - Samples:", nrow(modeling_data), "\n")

# ============================================================================
# 10. BUILDING CLASSIFICATION MODELS
# ============================================================================

cat("\n=== BUILDING CLASSIFICATION MODELS ===\n\n")

# Split data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(modeling_data$category, p = 0.7, list = FALSE)
train_data <- modeling_data[trainIndex, ]
test_data <- modeling_data[-trainIndex, ]

cat("✓ Training set:", nrow(train_data), "samples\n")
cat("✓ Testing set:", nrow(test_data), "samples\n")

# Model 1: Naive Bayes
cat("\n--- Training Naive Bayes Model ---\n")
model_nb <- naiveBayes(category ~ ., data = train_data)
pred_nb <- predict(model_nb, test_data)

# Model 2: Random Forest
cat("--- Training Random Forest Model ---\n")
model_rf <- randomForest(
  category ~ ., 
  data = train_data,
  ntree = 100,
  importance = TRUE
)
pred_rf <- predict(model_rf, test_data)

cat("\n✓ Models trained successfully\n")

# ============================================================================
# 11. MODEL EVALUATION
# ============================================================================

cat("\n=== MODEL EVALUATION ===\n\n")

# Confusion Matrix - Naive Bayes
cat("--- Naive Bayes Results ---\n")
cm_nb <- confusionMatrix(pred_nb, test_data$category)
print(cm_nb$overall)
print(cm_nb$byClass)

# Confusion Matrix - Random Forest
cat("\n--- Random Forest Results ---\n")
cm_rf <- confusionMatrix(pred_rf, test_data$category)
print(cm_rf$overall)
print(cm_rf$byClass)

# Per-class performance
cat("\n✓ Per-Class Balanced Accuracy:\n")
class_accuracy <- cm_rf$byClass[, "Balanced Accuracy"]
print(class_accuracy)

# Feature importance (Random Forest)
cat("\n✓ Top 10 Most Important Features:\n")
importance_scores <- importance(model_rf)
top_features <- head(importance_scores[order(-importance_scores[, "MeanDecreaseGini"]), ], 10)
print(top_features)

# ============================================================================
# 12. ROC CURVE ANALYSIS (One-vs-Rest)
# ============================================================================

cat("\n=== ROC CURVE ANALYSIS ===\n\n")

# Get probability predictions for Random Forest
pred_probs <- predict(model_rf, test_data, type = "prob")

# Calculate ROC for each class (One-vs-Rest)
roc_curves <- list()
auc_scores <- c()

for (class_name in levels(test_data$category)) {
  # Binary outcome: this class vs all others
  binary_outcome <- ifelse(test_data$category == class_name, 1, 0)
  
  # Check if class has at least one positive and one negative sample
  if (sum(binary_outcome == 1) > 0 && sum(binary_outcome == 0) > 0) {
    tryCatch({
      # ROC curve
      roc_obj <- roc(binary_outcome, pred_probs[, class_name], quiet = TRUE)
      roc_curves[[class_name]] <- roc_obj
      auc_scores[class_name] <- auc(roc_obj)
      
      cat("✓", class_name, "- AUC:", round(auc(roc_obj), 3), "\n")
    }, error = function(e) {
      cat("⚠ Skipping", class_name, "- insufficient data for ROC\n")
    })
  } else {
    cat("⚠ Skipping", class_name, "- no samples in test set\n")
  }
}

# Plot ROC curves
if (length(roc_curves) > 0) {
  cat("\n✓ Generating ROC curve plots...\n")
  
  # Create ROC plot
  plot(roc_curves[[1]], col = 1, main = "ROC Curves - One-vs-Rest Multi-Class Classification",
       lwd = 2)
  if (length(roc_curves) > 1) {
    for (i in 2:length(roc_curves)) {
      plot(roc_curves[[i]], col = i, add = TRUE, lwd = 2)
    }
  }
  legend("bottomright", 
         legend = paste(names(roc_curves), "AUC =", round(auc_scores[names(roc_curves)], 3)),
         col = 1:length(roc_curves), 
         lwd = 2,
         cex = 0.8)
} else {
  cat("\n⚠ Not enough data to generate ROC curves\n")
}

# ============================================================================
# 13. FINAL SUMMARY & INSIGHTS
# ============================================================================

cat("\n\n" , "=" %>% rep(60) %>% paste(collapse = ""), "\n")
cat("PROJECT SUMMARY - CUSTOMER SUPPORT TICKET CLASSIFIER\n")
cat("=" %>% rep(60) %>% paste(collapse = ""), "\n\n")

cat("1. DATA COLLECTION:\n")
cat("   ✓ Scraped", nrow(all_tickets), "real tickets from multiple sources\n")
cat("   ✓ Sources: Reddit, StackOverflow, HackerNews\n\n")

cat("2. TEXT PREPROCESSING:\n")
cat("   ✓ Cleaned and normalized text using pipe operators\n")
cat("   ✓ Removed HTML entities and special characters\n")
cat("   ✓ Extracted patterns: error codes, products, versions\n")
cat("   ✓ Created", nrow(tokens_clean), "clean tokens\n\n")

cat("3. TEXT MINING:\n")
cat("   ✓ Generated bigrams and trigrams\n")
cat("   ✓ Applied stemming and lemmatization\n")
cat("   ✓ Built Bag-of-Words with", ncol(bow_df), "features\n\n")

cat("4. CLASSIFICATION:\n")
cat("   ✓ Categories: bug, feature_request, query\n")
cat("   ✓ Trained Naive Bayes and Random Forest models\n")
cat("   ✓ Best model accuracy:", round(max(cm_rf$overall["Accuracy"]), 3), "\n\n")

cat("5. EVALUATION:\n")
cat("   ✓ Generated confusion matrices\n")
cat("   ✓ Calculated per-class metrics\n")
if (length(roc_curves) > 0) {
  cat("   ✓ Created One-vs-Rest ROC curves\n")
  cat("   ✓ Mean AUC:", round(mean(auc_scores, na.rm = TRUE), 3), "\n\n")
} else {
  cat("   ⚠ ROC curves skipped due to insufficient test data\n\n")
}

cat("=" %>% rep(60) %>% paste(collapse = ""), "\n")
cat("✓ PROJECT COMPLETED SUCCESSFULLY!\n")
cat("=" %>% rep(60) %>% paste(collapse = ""), "\n\n")

# Save results
saveRDS(all_tickets, "ticket_data.rds")
saveRDS(model_rf, "ticket_classifier_model.rds")
cat("✓ Data and model saved to disk\n")