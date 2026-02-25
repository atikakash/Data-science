pkgs <- c(
  "dplyr", "stringr", "tidytext", "tidyr", "tibble",
  "stopwords", "SnowballC", "tm"
)

to_install <- pkgs[!pkgs %in% installed.packages()[, "Package"]]
if (length(to_install) > 0) install.packages(to_install)

library(dplyr)
library(stringr)
library(tidytext)
library(tidyr)
library(tibble)
library(stopwords)
library(SnowballC)
library(tm)

data1 <- read.csv("C:/Dataset/ml_dl_papers.csv", header = TRUE)
data2 <- read.csv("C:/Dataset/non_ai_non_ml_papers.csv", header = TRUE)


pick_text_col <- function(df) {
  candidates <- c("abstract", "Abstract", "ABSTRACT", "title", "Title", "TEXT", "text")
  found <- candidates[candidates %in% names(df)]
  if (length(found) > 0) return(found[1])
  return(names(df)[1])
}

text_col1 <- pick_text_col(data1)
text_col2 <- pick_text_col(data2)

text1 <- as.character(data1[[text_col1]])
text2 <- as.character(data2[[text_col2]])

text1 <- text1[!is.na(text1)]
text2 <- text2[!is.na(text2)]

clean_text <- function(x) {
  x <- tolower(x)
  x <- gsub("<[^>]+>", " ", x)
  x <- gsub("\\bcan't\\b", "cannot", x)
  x <- gsub("\\bwon't\\b", "will not", x)
  x <- gsub("\\bi'm\\b", "i am", x)
  x <- gsub("\\bit's\\b", "it is", x)
  x <- gsub("\\bdon't\\b", "do not", x)
  x <- gsub("\\bdoesn't\\b", "does not", x)
  x <- iconv(x, from = "", to = "ASCII//TRANSLIT", sub = " ")
  x <- gsub("[[:punct:]]", " ", x)
  x <- gsub("[0-9]+", " ", x)
  x <- gsub("[^a-z\\s]", " ", x)
  x <- gsub("\\s+", " ", x)
  x <- trimws(x)
  
  return(x)
}

text1_clean <- clean_text(text1)
text2_clean <- clean_text(text2)

df <- tibble(
  doc_id = c(paste0("ML_", seq_along(text1_clean)),
             paste0("NONML_", seq_along(text2_clean))),
  class  = c(rep("ML", length(text1_clean)),
             rep("NONML", length(text2_clean))),
  text   = c(text1_clean, text2_clean)
)

tidy_tokens <- df %>%
  unnest_tokens(word, text) %>%
  filter(!word %in% stopwords("en")) %>%
  filter(nchar(word) >= 2)

tf_idf <- tidy_tokens %>%
  count(doc_id, word, sort = TRUE) %>%
  bind_tf_idf(word, doc_id, n)

top_terms <- tf_idf %>%
  arrange(desc(tf_idf)) %>%
  slice_head(n = 20)

print(top_terms)

tfidf_dtm <- tf_idf %>%
  cast_dtm(doc_id, word, tf_idf)

tfidf_dtm

tfidf_mat <- as.matrix(tfidf_dtm)
row_norm <- sqrt(rowSums(tfidf_mat^2))
row_norm[row_norm == 0] <- 1
tfidf_mat_norm <- tfidf_mat / row_norm

tfidf_class <- tidy_tokens %>%
  count(class, word) %>%
  bind_tf_idf(word, class, n)

contrastive_features <- tfidf_class %>%
  select(class, word, tf_idf) %>%
  pivot_wider(names_from = class, values_from = tf_idf, values_fill = 0) %>%
  mutate(contrast_score = ML - NONML) %>%
  arrange(desc(abs(contrast_score)))

print(head(contrastive_features, 20))

corpus <- VCorpus(VectorSource(df$text))

dtm_tm <- DocumentTermMatrix(
  corpus,
  control = list(
    wordLengths = c(2, Inf),
    stopwords = TRUE,
    weighting = weightTfIdf
  )
)

inspect(dtm_tm[1:5, 1:5])

pkgs <- c("dplyr","tibble","tidyr","ggplot2","factoextra","cluster","proxy",
          "tm","mclust","wordcloud")

to_install <- pkgs[!pkgs %in% installed.packages()[,"Package"]]
if(length(to_install) > 0) install.packages(to_install)

library(dplyr)
library(tibble)
library(tidyr)
library(ggplot2)
library(factoextra)
library(cluster)
library(proxy)
library(tm)
library(mclust)
library(wordcloud)

tfidf_mat <- as.matrix(tfidf_dtm)

common_docs <- intersect(rownames(tfidf_mat), df$doc_id)
tfidf_mat <- tfidf_mat[common_docs, , drop = FALSE]

tfidf_mat <- tfidf_mat[rowSums(tfidf_mat) > 0, , drop = FALSE]
tfidf_mat <- tfidf_mat[, colSums(tfidf_mat) > 0, drop = FALSE]

row_norm <- sqrt(rowSums(tfidf_mat^2))
row_norm[row_norm == 0] <- 1
tfidf_mat_norm <- tfidf_mat / row_norm

df_aligned <- df %>%
  filter(doc_id %in% rownames(tfidf_mat_norm)) %>%
  mutate(doc_id = factor(doc_id, levels = rownames(tfidf_mat_norm))) %>%
  arrange(doc_id)

set.seed(123)

fviz_nbclust(tfidf_mat_norm, kmeans, method = "wss") +
  ggtitle("Elbow Method (WSS)")

fviz_nbclust(tfidf_mat_norm, kmeans, method = "silhouette") +
  ggtitle("Silhouette Method")
k <- 2

set.seed(123)
km <- kmeans(tfidf_mat_norm, centers = k, nstart = 25)

km_clusters <- km$cluster
table(km_clusters)

fviz_cluster(list(data = tfidf_mat_norm, cluster = km_clusters)) +
  ggtitle("K-means Clusters (PCA Projection)")
 
d_cos <- proxy::dist(tfidf_mat_norm, method = "cosine")
hc <- hclust(as.dist(d_cos), method = "average")

plot(hc, cex = 0.6, main = "Hierarchical Clustering (Cosine) Dendrogram")

hc_clusters <- cutree(hc, k = k)
table(hc_clusters)
 
fviz_dend(hc, k = k, rect = TRUE, show_labels = FALSE) +
  ggtitle("Dendrogram with Cluster Cuts")

get_top_words_per_cluster <- function(mat, clusters, top_n = 10){
  mat_df <- as.data.frame(mat)
  mat_df$cluster <- clusters
  
  centroids <- mat_df %>%
    group_by(cluster) %>%
    summarise(across(where(is.numeric), mean), .groups = "drop")
  
  long <- centroids %>%
    pivot_longer(cols = -cluster, names_to = "word", values_to = "score")
  
  top_words <- long %>%
    group_by(cluster) %>%
    arrange(desc(score)) %>%
    slice_head(n = top_n) %>%
    ungroup()
  
  top_words
}

top_km_words <- get_top_words_per_cluster(tfidf_mat, km_clusters, top_n = 10)
top_hc_words <- get_top_words_per_cluster(tfidf_mat, hc_clusters, top_n = 10)

top_km_words
top_hc_words

ggplot(top_km_words, aes(x = reorder(word, score), y = score)) +
  geom_col() +
  coord_flip() +
  facet_wrap(~ cluster, scales = "free_y") +
  labs(title = "Top 10 Words per Cluster (K-means)",
       x = "Word", y = "Average TF-IDF") +
  theme_minimal()

ggplot(top_hc_words, aes(x = reorder(word, score), y = score)) +
  geom_col() +
  coord_flip() +
  facet_wrap(~ cluster, scales = "free_y") +
  labs(title = "Top 10 Words per Cluster (Hierarchical)",
       x = "Word", y = "Average TF-IDF") +
  theme_minimal()

par(mfrow = c(1, k))
for (cl in sort(unique(km_clusters))) {
  words <- top_km_words %>% filter(cluster == cl)
  wordcloud(words = words$word, freq = words$score,
            max.words = 10, random.order = FALSE,
            main = paste("K-means Cluster", cl))
}
par(mfrow = c(1,1))

pca <- prcomp(tfidf_mat_norm, center = TRUE, scale. = FALSE)
pca_df <- data.frame(pca$x[,1:2], doc_id = rownames(tfidf_mat_norm))
pca_df$kmeans_cluster <- factor(km_clusters[match(pca_df$doc_id, names(km_clusters))])

if ("class" %in% names(df_aligned)) {
  pca_df$class <- df_aligned$class
  ggplot(pca_df, aes(PC1, PC2, color = kmeans_cluster, shape = class)) +
    geom_point(alpha = 0.8, size = 2) +
    theme_minimal() +
    labs(title = "PCA Plot (Color=Cluster, Shape=Class)")
} else {
  ggplot(pca_df, aes(PC1, PC2, color = kmeans_cluster)) +
    geom_point(alpha = 0.8, size = 2) +
    theme_minimal() +
    labs(title = "PCA Plot (Color=Cluster)")
}

cat("K-means cluster sizes:\n")
print(table(km_clusters))

cat("\nHierarchical cluster sizes:\n")
print(table(hc_clusters))

sil_km <- silhouette(km_clusters, dist(tfidf_mat_norm))
sil_hc <- silhouette(hc_clusters, dist(tfidf_mat_norm))

cat("Average silhouette (K-means):", mean(sil_km[,3]), "\n")
cat("Average silhouette (Hierarchical):", mean(sil_hc[,3]), "\n")

fviz_silhouette(sil_km) + ggtitle("Silhouette Plot - K-means")
fviz_silhouette(sil_hc) + ggtitle("Silhouette Plot - Hierarchical")

if ("class" %in% names(df_aligned)) {
  doc_order <- rownames(tfidf_mat_norm)
  true_class <- df_aligned$class[match(doc_order, df_aligned$doc_id)]
  
  cat("\nK-means vs True Class:\n")
  print(table(True = true_class, Cluster = km_clusters))
  
  cat("\nHierarchical vs True Class:\n")
  print(table(True = true_class, Cluster = hc_clusters))
  
  cat("\nAdjusted Rand Index (K-means): ",
      adjustedRandIndex(as.numeric(factor(true_class)), km_clusters), "\n")
  
  cat("Adjusted Rand Index (Hierarchical): ",
      adjustedRandIndex(as.numeric(factor(true_class)), hc_clusters), "\n")
}
