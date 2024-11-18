#===========================1. Load libraries===========================
library(tidyverse)
library(tidytext)
library(text2vec)
library(textdata)
library(textclean)
library(stringr)
library(caret)
library(assertthat)
library(glmnet)
library(vip)
library(tm)
library(quanteda)
library(SnowballC)
library(randomForest)
library(ranger)
library(xgboost)
library(ROCR)
library(gbm)
#install.packages("geosphere")
library(geosphere)

#===========================2. Load datasets===========================
setwd('~/Downloads/')
train_x <- read_csv("airbnb_train_x_2024.csv")
train_y <- read_csv("airbnb_train_y_2024.csv")
test_x <- read_csv("airbnb_test_x_2024.csv")
# External dataset - airport
airport <- read_csv("airport-codes.csv")

# Join the training y to the training x file, and turn the target variables into factors
train <- cbind(train_x, train_y) %>%
  mutate(perfect_rating_score = as.factor(perfect_rating_score),
         high_booking_rate = as.factor(high_booking_rate)) 

summary(train)
names(train) 

# Data Cleaning in "airport" files: remove closed and heliport
airport$type <- as.factor(airport$type)
airport <- airport %>%
  filter((type %in% c('large_airport', 'medium_airport')) & is.na(continent))

#===========================3. Data Cleaning===========================
#===========================3-1. Data Cleaning: Training data===========================
# Create "train_cleaned" to store cleaned training data
train_cleaned = train
# Clean the existing data
train_cleaned <- train_cleaned %>%
  mutate(
    bedrooms = ifelse(is.na(bedrooms), 0, bedrooms),
    beds = ifelse(is.na(beds), 0, beds),
    host_total_listings_count = ifelse(is.na(host_total_listings_count), 1, host_total_listings_count),
    accommodates = ifelse(is.na(accommodates), 0, accommodates),
    room_type = as.factor(room_type),
    square_feet = ifelse(is.na(square_feet), 0, square_feet),
    security_deposit = ifelse(is.na(security_deposit), 0, security_deposit),
    cleaning_fee = ifelse(is.na(cleaning_fee), 0, cleaning_fee),
    price = ifelse(is.na(price), mean(price, na.rm = TRUE), price),
    bathrooms = ifelse(is.na(bathrooms), median(bathrooms, na.rm = TRUE), bathrooms))

# Create and clean new variables

# has_security_deposit, charges_for_extra, has_cleaning_fee
train_cleaned <- train_cleaned %>%
  mutate(has_security_deposit = as.factor(ifelse(security_deposit != 0, "YES", "NO")),
         charges_for_extra = as.factor(ifelse(extra_people > 0, "YES", "NO")),
         has_cleaning_fee = as.factor(ifelse(cleaning_fee != 0, "YES", "NO")))


# cancellation_policy
train_cleaned$cancellation_policy <- as.factor(ifelse(train_cleaned$cancellation_policy %in% 
                                              c("strict", "super_strict_30","super_strict_60", "no_refunds"), "strict", train_cleaned$cancellation_policy))

# bed category
train_cleaned$bed_category <- as.factor(ifelse(train_cleaned$bed_type == "Real Bed", "bed", "other"))

# property_type
train_cleaned <- train_cleaned %>%
  mutate(property_category = case_when(
    property_type %in% c("Apartment", "Serviced apartment", "Loft") ~ "apartment",
    property_type %in% c("Bed & Breakfast", "Boutique hotel", "Hostel") ~ "hotel",
    property_type %in% c("Townhouse", "Condominium") ~ "condo",
    property_type %in% c("Bungalow", "House") ~ "house",
    TRUE ~ "other"
  )) %>%
  mutate(property_category = as.factor(property_category))

# ppp_ind
train_cleaned$price_per_person <- train_cleaned$price / train_cleaned$accommodates
train_cleaned <- train_cleaned %>%
  group_by(property_category) %>%
  mutate(ppp_median = median(price_per_person),
         ppp_ind = as.factor(ifelse(price_per_person > ppp_median, 1, 0))) %>%
  ungroup() %>%
  select(-ppp_median)

# host_acceptance
train_cleaned <- train_cleaned %>%
  mutate(host_acceptance = case_when(
    host_acceptance_rate == "100%" ~ "ALL",
    is.na(host_acceptance_rate) ~ "MISSING",
    TRUE ~ "SOME"
  )) %>%
  mutate(host_response = as.factor(host_acceptance))

# host_response
train_cleaned$host_response_rate<- as.character(train_cleaned$host_response_rate)
train_cleaned <- train_cleaned %>%
  mutate(host_response = case_when(
    host_response_rate == "100" ~ "ALL",
    is.na(host_response_rate) ~ "MISSING",
    TRUE ~ "SOME")) %>%
  mutate(host_response = as.factor(host_response))

# has_min_nights, max_nights_group
train_cleaned$has_min_nights <- as.factor(ifelse(train_cleaned$minimum_nights > 1, "YES", "NO"))
train_cleaned <- train_cleaned %>%
  mutate(
    max_nights_group = case_when(
      maximum_nights <= 30 ~ "Short-term",
      maximum_nights > 30 & maximum_nights <= 90 ~ "Medium-term",
      maximum_nights > 90 & maximum_nights <= 365 ~ "Long-term",
      maximum_nights > 365 & maximum_nights <= 1825 ~ "Very long-term",
      TRUE ~ "No limit")) %>%
  mutate(max_nights_group = as.factor(max_nights_group))

# has_license
train_cleaned$has_license <- ifelse(is.na(train_cleaned$license), "NO", train_cleaned$license)
train_cleaned$has_license <- ifelse(train_cleaned$license %in% c("City registration pending", "CITY REGISTRATION PENDING"), "Pending", "Yes")
train_cleaned$has_license = as.factor(train_cleaned$has_license)

# market
train_cleaned <- train_cleaned %>%
  mutate(market = case_when(
    market %in% c("Austin") ~ "Austin",
    market %in% c ("Chicago") ~ "Chicago",
    market %in% c("D.C.") ~ "D.C.",
    market %in% c("Los Angeles") ~ "Los Angeles",
    market %in% c("New Orleans") ~ "New Orleans",
    market %in% c("New York") ~ "New York",
    market %in% c("San Diego") ~ "San Diego",
    market %in% c("San Francisco") ~ "San Francisco",
    market %in% c("Seattle") ~ "Seattle",
    TRUE ~ "OTHER"
  )) %>%
  mutate(market = as.factor(market))

# availability_rate_30, availability_rate_60, availability_rate_90
train_cleaned <- train_cleaned %>%
  mutate(availability_rate_30 = availability_30/30,
         availability_rate_60 = availability_60/60,
         availability_rate_90 = availability_90/90,
         availability_rate_365 = availability_365/365) 

# host_year(assume the data is in 2020)
train_cleaned <- train_cleaned %>% 
  mutate(host_year = 2020 - as.numeric(format(as.Date(host_since), "%Y")),
         host_year = ifelse(is.na(host_year), 0, host_year))

# dummy variables from "features" variable
train_cleaned <- train_cleaned %>% 
  mutate(
    super_host = as.factor(ifelse(grepl("Host Is Superhost", features) , 1, 0)),
    has_profile_pic = as.factor(ifelse(grepl("Host Has Profile Pic", features) , 1, 0)),
    identity_verified = as.factor(ifelse(grepl("Host Identity Verified", features) , 1, 0)),
    location_exact = as.factor(ifelse(grepl("Is Location Exact", features) , 1, 0)),
    instant_bookable = as.factor(ifelse(grepl("Instant Bookable", features) , 1, 0)),
    requires_license = as.factor(ifelse(grepl("Requires License", features) , 1, 0)),
    guest_profile_pic_required = as.factor(ifelse(grepl("Require Guest Profile Picture", features) , 1, 0)),
    guest_phone_verification_required = as.factor(ifelse(grepl("Require Guest Phone Verification", features) , 1, 0)))

# jurisdiction_category
train_cleaned <- train_cleaned %>%
  mutate(jurisdiction_category = case_when(
    jurisdiction_names %in% c("DISTRICT OF COLUMBIA, WASHINGTON", "WASHINGTON", "DISTRICT OF COLUMBIA, STATE DEPARTMENT", "Montgomery County, MD") ~ "DC area",
    jurisdiction_names %in% c("OREGON, MULTNOMAH, PORTLAND", "OREGON", "OREGON, PORTLAND", "OREGON, Washington County, OR", "OREGON, MULTNOMAH", "OREGON, Lane County, OR, Eugene, OR") ~ "Oregon",
    jurisdiction_names %in% c("SAN FRANCISCO", "SAN DIEGO, SAN DIEGO TOURISM MARKETING DISTRICT A, SAN DIEGO TOURISM MARKETING DISTRICT B", "City of Los Angeles, CA", "MALIBU", "Santa Monica", "PALO ALTO", "OAKLAND") ~ "California",
    jurisdiction_names %in% c("Illinois State, Cook County, IL, CHICAGO", "Illinois State, Cook County, IL", "Illinois State, Cook County, IL, Oak Park, IL") ~ "Chicago",
    jurisdiction_names %in% c("Louisiana State","Louisiana State, New Orleans, LA") ~ "Louisiana",
    TRUE ~ "Unknown"
  )) %>%
  mutate(jurisdiction_category = as.factor(jurisdiction_category))

# Discount: has_weekly_discount, has_monthly_discount
train_cleaned <- train_cleaned %>%
  mutate(
    expected_weekly_price = 7 * price,
    expected_monthly_price = 30.4 * price,
    has_weekly_discount = ifelse(!is.na(weekly_price) & weekly_price < expected_weekly_price,
                                 "YES", "NO"),
    has_monthly_discount = ifelse(!is.na(monthly_price) & monthly_price < expected_monthly_price,
                                  "YES", "NO")) %>%
  mutate(has_weekly_discount = as.factor(has_weekly_discount),
         has_monthly_discount = as.factor(has_monthly_discount))

# amenities
# handle amenities
# create tokenizer
cleaning_tokenizer_amenities <- function(v) {
  v %>%
    space_tokenizer(sep = ',')}
# tokenize
it_amenities <- itoken(train_cleaned$amenities, 
                   preprocessor = tolower,
                   tokenizer = cleaning_tokenizer_amenities,
                   progressbar = FALSE)
# learn the vocabulary
vocab_amenities <- create_vocabulary(it_amenities)
# vectorize
vectorizer_amenities <- vocab_vectorizer(vocab_amenities)
dtm_amenities <- create_dtm(it_amenities, vectorizer_amenities)
# create a datarame
amenities_matrix <- data.frame(as.matrix(dtm_amenities))
# combine train_cleaned and amenities_matrix
train_cleaned <- cbind(train_cleaned, amenities_matrix)

# nearby_airport(distance within 30km): using the external dataset-airport.csv
distance_to_airport <- sapply(1:nrow(train_cleaned), function(i) {
  min(sapply(1:nrow(airport), function(j) {
    distHaversine(c(train_cleaned$longitude[i], train_cleaned$latitude[i]), c(airport$Long[j], airport$Lat[j]))
  }))
})
train_cleaned <- cbind(train_cleaned, distance_to_airport/1000)
train_cleaned <- train_cleaned %>%
  mutate(nearby_airport=as.factor(ifelse(distance_to_airport/1000<=30,1,0)))

# nearby_attractions
train_cleaned <- train_cleaned %>%
  mutate(nearby_attractions = as.factor(ifelse(!is.na(host_neighbourhood), 1, 0)))

# replace other numerical NA values with mean
train_cleaned <- train_cleaned %>%
  mutate_if(is.numeric, ~replace_na(., mean(., na.rm = TRUE)))

#===========================3-2. Data Cleaning: Test data===========================
# Create "train_cleaned" to store cleaned training data
test_clean = test_x
# Clean the existing data
test_clean <- test_clean %>%
  mutate(
    bedrooms = ifelse(is.na(bedrooms), 0, bedrooms),
    beds = ifelse(is.na(beds), 0, beds),
    host_total_listings_count = ifelse(is.na(host_total_listings_count), 1, host_total_listings_count),
    accommodates = ifelse(is.na(accommodates), 0, accommodates),
    room_type = as.factor(room_type),
    square_feet = ifelse(is.na(square_feet), 0, square_feet),
    security_deposit = ifelse(is.na(security_deposit), 0, security_deposit),
    cleaning_fee = ifelse(is.na(cleaning_fee), 0, cleaning_fee),
    price = ifelse(is.na(price), mean(price, na.rm = TRUE), price),
    bathrooms = ifelse(is.na(bathrooms), median(bathrooms, na.rm = TRUE), bathrooms))

# has_security_deposit, charges_for_extra, has_cleaning_fee
test_clean <- test_clean %>%
  mutate(has_security_deposit = as.factor(ifelse(security_deposit != 0, "YES", "NO")),
         charges_for_extra = as.factor(ifelse(extra_people > 0, "YES", "NO")),
         has_cleaning_fee = as.factor(ifelse(cleaning_fee != 0, "YES", "NO")))

# cancellation_policy
test_clean$cancellation_policy <- as.factor(ifelse(test_clean$cancellation_policy %in% 
                                                        c("strict", "super_strict_30","super_strict_60", "no_refunds"), "strict", test_clean$cancellation_policy))

# bed category
test_clean$bed_category <- as.factor(ifelse(test_clean$bed_type == "Real Bed", "bed", "other"))

# property_type
test_clean <- test_clean %>%
  mutate(property_category = case_when(
    property_type %in% c("Apartment", "Serviced apartment", "Loft") ~ "apartment",
    property_type %in% c("Bed & Breakfast", "Boutique hotel", "Hostel") ~ "hotel",
    property_type %in% c("Townhouse", "Condominium") ~ "condo",
    property_type %in% c("Bungalow", "House") ~ "house",
    TRUE ~ "other"
  )) %>%
  mutate(property_category = as.factor(property_category))

# ppp_ind
test_clean$price_per_person <- test_clean$price / test_clean$accommodates
test_clean <- test_clean %>%
  group_by(property_category) %>%
  mutate(ppp_median = median(price_per_person),
         ppp_ind = as.factor(ifelse(price_per_person > ppp_median, 1, 0))) %>%
  ungroup() %>%
  select(-ppp_median)

# host_acceptance
test_clean <- test_clean %>%
  mutate(host_acceptance = case_when(
    host_acceptance_rate == "100%" ~ "ALL",
    is.na(host_acceptance_rate) ~ "MISSING",
    TRUE ~ "SOME"
  )) %>%
  mutate(host_response = as.factor(host_acceptance))

# host_response
test_clean$host_response_rate<- as.character(test_clean$host_response_rate)
test_clean <- test_clean %>%
  mutate(host_response = case_when(
    host_response_rate == "100" ~ "ALL",
    is.na(host_response_rate) ~ "MISSING",
    TRUE ~ "SOME")) %>%
  mutate(host_response = as.factor(host_response))

# has_min_nights, max_nights_group
test_clean$has_min_nights <- as.factor(ifelse(test_clean$minimum_nights > 1, "YES", "NO"))
test_clean <- test_clean %>%
  mutate(
    max_nights_group = case_when(
      maximum_nights <= 30 ~ "Short-term",
      maximum_nights > 30 & maximum_nights <= 90 ~ "Medium-term",
      maximum_nights > 90 & maximum_nights <= 365 ~ "Long-term",
      maximum_nights > 365 & maximum_nights <= 1825 ~ "Very long-term",
      TRUE ~ "No limit")) %>%
  mutate(max_nights_group = as.factor(max_nights_group))

# has_license
test_clean$has_license <- ifelse(is.na(test_clean$license), "NO", test_clean$license)
test_clean$has_license <- ifelse(test_clean$license %in% c("City registration pending", "CITY REGISTRATION PENDING"), "Pending", "Yes")
test_clean$has_license = as.factor(test_clean$has_license)

# market
test_clean <- test_clean %>%
  mutate(market = case_when(
    market %in% c("Austin") ~ "Austin",
    market %in% c ("Chicago") ~ "Chicago",
    market %in% c("D.C.") ~ "D.C.",
    market %in% c("Los Angeles") ~ "Los Angeles",
    market %in% c("New Orleans") ~ "New Orleans",
    market %in% c("New York") ~ "New York",
    market %in% c("San Diego") ~ "San Diego",
    market %in% c("San Francisco") ~ "San Francisco",
    market %in% c("Seattle") ~ "Seattle",
    TRUE ~ "OTHER"
  )) %>%
  mutate(market = as.factor(market))

# availability_rate_30, availability_rate_60, availability_rate_90
test_clean <- test_clean %>%
  mutate(availability_rate_30 = availability_30/30,
         availability_rate_60 = availability_60/60,
         availability_rate_90 = availability_90/90,
         availability_rate_365 = availability_365/365) 

# host_year(assume the data is in 2020)
test_clean <- test_clean %>% 
  mutate(host_year = 2020 - as.numeric(format(as.Date(host_since), "%Y")),
         host_year = ifelse(is.na(host_year), 0, host_year))

# dummy variables from "features" variable
test_clean <- test_clean %>% 
  mutate(
    super_host = as.factor(ifelse(grepl("Host Is Superhost", features) , 1, 0)),
    has_profile_pic = as.factor(ifelse(grepl("Host Has Profile Pic", features) , 1, 0)),
    identity_verified = as.factor(ifelse(grepl("Host Identity Verified", features) , 1, 0)),
    location_exact = as.factor(ifelse(grepl("Is Location Exact", features) , 1, 0)),
    instant_bookable = as.factor(ifelse(grepl("Instant Bookable", features) , 1, 0)),
    requires_license = as.factor(ifelse(grepl("Requires License", features) , 1, 0)),
    guest_profile_pic_required = as.factor(ifelse(grepl("Require Guest Profile Picture", features) , 1, 0)),
    guest_phone_verification_required = as.factor(ifelse(grepl("Require Guest Phone Verification", features) , 1, 0)))

# jurisdiction_category
test_clean <- test_clean %>%
  mutate(jurisdiction_category = case_when(
    jurisdiction_names %in% c("DISTRICT OF COLUMBIA, WASHINGTON", "WASHINGTON", "DISTRICT OF COLUMBIA, STATE DEPARTMENT", "Montgomery County, MD") ~ "DC area",
    jurisdiction_names %in% c("OREGON, MULTNOMAH, PORTLAND", "OREGON", "OREGON, PORTLAND", "OREGON, Washington County, OR", "OREGON, MULTNOMAH", "OREGON, Lane County, OR, Eugene, OR") ~ "Oregon",
    jurisdiction_names %in% c("SAN FRANCISCO", "SAN DIEGO, SAN DIEGO TOURISM MARKETING DISTRICT A, SAN DIEGO TOURISM MARKETING DISTRICT B", "City of Los Angeles, CA", "MALIBU", "Santa Monica", "PALO ALTO", "OAKLAND") ~ "California",
    jurisdiction_names %in% c("Illinois State, Cook County, IL, CHICAGO", "Illinois State, Cook County, IL", "Illinois State, Cook County, IL, Oak Park, IL") ~ "Chicago",
    jurisdiction_names %in% c("Louisiana State","Louisiana State, New Orleans, LA") ~ "Louisiana",
    TRUE ~ "Unknown"
  )) %>%
  mutate(jurisdiction_category = as.factor(jurisdiction_category))

# Discount: has_weekly_discount, has_monthly_discount
test_clean <- test_clean %>%
  mutate(
    expected_weekly_price = 7 * price,
    expected_monthly_price = 30.4 * price,
    has_weekly_discount = ifelse(!is.na(weekly_price) & weekly_price < expected_weekly_price, "YES", "NO"),
    has_monthly_discount = ifelse(!is.na(monthly_price) & monthly_price < expected_monthly_price, "YES", "NO")) %>%
  mutate(has_weekly_discount = as.factor(has_weekly_discount),
         has_monthly_discount = as.factor(has_monthly_discount))

# amenities
# handle amenities
# tokenize
it_amenities_test <- itoken(test_clean$amenities, 
                            preprocessor = tolower,
                            tokenizer = cleaning_tokenizer_amenities,
                            progressbar = FALSE)

# vectorize
dtm_amenities_test <- create_dtm(it_amenities_test, vectorizer_amenities)
# create a datarame
amenities_test_matrix <- data.frame(as.matrix(dtm_amenities_test))
# combine two df
test_clean <- cbind(test_clean, amenities_test_matrix)

# nearby_airport(distance within 30km): using the external dataset-airport.csv
distance_to_airport_test <- sapply(1:nrow(test_clean), function(i) {
  min(sapply(1:nrow(airport), function(j) {
    distHaversine(c(test_clean$longitude[i], test_clean$latitude[i]), c(airport$Long[j], airport$Lat[j]))
  }))
})
test_clean <- cbind(test_clean, distance_to_airport_test/1000)
test_clean <- test_clean %>%
  mutate(nearby_airport=as.factor(ifelse(distance_to_airport_test/1000<=30,1,0)))

# nearby_attractions
test_clean <- test_clean %>%
  mutate(nearby_attractions = as.factor(ifelse(!is.na(host_neighbourhood), 1, 0)))

#===========================4. Text mining: EDA & create new variables===========================
#===========================4-1. Training set: preparation===========================
# Training set
# Create tokenizer
cleaning_tokenizer <- function(v) {
  v %>%
    removeNumbers %>% #remove all numbers
    removePunctuation %>% #remove all punctuation
    removeWords(tm::stopwords(kind="en")) %>% #remove stopwords
    stemDocument %>%
    word_tokenizer}
# add some other stop_words
stop_words <- c("the", "end", "to", "is", "a", "and", "_")

#===========================4-2. Text mining: training set - "summary"===========================
# EDA & new variable: "positive_score" ---------------------
# EDA on "summary" by using ridge model
# itoken "summary"
it_summary <- itoken(train_cleaned$summary, 
                         preprocessor = tolower, #preprocessing by converting to lowercase
                         tokenizer = cleaning_tokenizer, 
                         progressbar = FALSE)

vocab_summary <- create_vocabulary(it_summary, stopwords = stop_words, ngram = c(1L, 2L)) %>%
  prune_vocabulary(term_count_min = 50, doc_proportion_max = 0.5)
# vectorizer
vectorizer_summary <- vocab_vectorizer(vocab_summary)
# convert the training documents into a DTM
dtm_train_summary <- create_dtm(it_summary, vectorizer_summary)
# count of each term in corpus
Matrix::colSums(dtm_train_summary)

# split dtm_train_summary into train and valid
set.seed(1)
tr_rows <- sample(nrow(train), 0.7*nrow(train))
tr_dtm_summary <- dtm_train_summary[tr_rows,]
va_dtm_summary <- dtm_train_summary[-tr_rows,]

# get the y values
tr_y <- train[tr_rows,]$high_booking_rate
va_y <- train[-tr_rows,]$high_booking_rate

# Build ridge model
ridge_model_summary <- glmnet(tr_dtm_summary, tr_y, family="binomial", alpha=0, lambda=.01)
pred_ridge_summary <- predict(ridge_model_summary, newx = va_dtm_summary, type="response")
# factors are assigned to 0-1 in alphabetical order. Here Android becomes 0 and iPhone becomes 1 in the linear model
class_ridge_summary <- ifelse(pred_ridge_summary > 0.5, "YES", "NO")
acc_ridge_summary <- mean(ifelse(class_ridge_summary == va_y, 1,0))
acc_ridge_summary

# Make a variable importance plot
vip(ridge_model_summary, num_features = 20)

# get the negative and positive sentiment words
bing_negative <- get_sentiments("bing") %>%
  filter(sentiment == 'negative')

bing_positive <- get_sentiments("bing") %>%
  filter(sentiment == 'positive')

# create a new variable: positive_score
# define positive and negative words
positive_words <- bing_positive$word
negative_words <- bing_negative$word

# function to count positive and negative words in each summary
count_sentiments <- function(tokens) {
  positive_count <- sum(tokens %in% positive_words)
  negative_count <- sum(tokens %in% negative_words)
  return(data.frame(positive_count = positive_count, negative_count = negative_count))
}

# apply the function to each summary
train_summary <- train_cleaned$summary
tokenized_summary <- lapply(strsplit(tolower(as.character(train_summary)), "\\s+"), function(tokens) {
  tokens <- tokens[tokens != ""]
  return(tokens)
})
sentiment_counts_summary <- lapply(tokenized_summary, count_sentiments)
# convert the list of data frames to a single data frame
sentiment_summary <- do.call(rbind, sentiment_counts_summary) %>%
  mutate(po_ne_diff = positive_count - negative_count,
         max_positive_score = max(po_ne_diff),
         positive_score = po_ne_diff/max_positive_score)

# add the new variables from text mining into train_cleaned
train_cleaned <- cbind(train_cleaned, positive_score = sentiment_summary$positive_score)

#===========================4-3. Text mining: training set - "transit"===========================
# EDA & new variable: "transit_score" ---------------------
# itoken "transit"
it_transit <- itoken(train_cleaned$transit, 
                         preprocessor = tolower, #preprocessing by converting to lowercase
                         tokenizer = cleaning_tokenizer, 
                         progressbar = FALSE)

vocab_transit <- create_vocabulary(it_transit, stopwords = stop_words, ngram = c(1L, 2L)) %>%
  prune_vocabulary(term_count_min = 50, doc_proportion_max=0.3)

# vectorizer
vectorizer_transit <- vocab_vectorizer(vocab_transit)

# convert the training documents into a DTM and make it a binary BOW matrix
dtm_transit <- create_dtm(it_transit, vectorizer_transit)

# split tr/va
set.seed(1)
tr_rows <- sample(nrow(train), 0.7*nrow(train))
tr_dtm_transit <- dtm_transit[tr_rows,]
va_dtm_transit <- dtm_transit[-tr_rows,]

# Get the y values
tr_y <- train[tr_rows,]$high_booking_rate
va_y <- train[-tr_rows,]$high_booking_rate

# build ridge model
ridge_model_transit <- glmnet(tr_dtm_transit, tr_y, family="binomial", alpha=0, lambda=.01)
pred_ridge_transit <- predict(ridge_model_transit, newx = va_dtm_transit, type="response")
class_ridge_transit <- ifelse(pred_ridge_transit > 0.5, "YES", "NO")
acc_ridge_transit <- mean(ifelse(class_ridge_transit == va_y, 1,0))
acc_ridge_transit

# Make a variable importance plot
vip(ridge_model_transit, num_features = 20)

# create a new variable: transit_score
# apply the "count_sentiments" function to each transit
train_transit <- train_cleaned$transit
tokenized_transit <- lapply(strsplit(tolower(as.character(train_transit)), "\\s+"), function(tokens) {
  # Remove empty tokens
  tokens <- tokens[tokens != ""]
  return(tokens)
})
sentiment_counts_transit <- lapply(tokenized_transit, count_sentiments)
# convert the list of data frames to a single data frame
sentiment_df_transit <- do.call(rbind, sentiment_counts_transit) %>%
  mutate(po_ne_diff = positive_count - negative_count)

# add the new variables from text mining into train_cleaned
train_cleaned <- cbind(train_cleaned, transit_diff = sentiment_df_transit$po_ne_diff) 
train_cleaned <- train_cleaned %>%
  group_by(market) %>%
  mutate(max_transit_score = max(transit_diff),
         transit_score = transit_diff/max_transit_score) %>%
  ungroup()

#===========================4-4. Text mining: training set - "interaction"===========================
# EDA & new variable: "interaction_score" ---------------------
# itoken "interation"
it_interaction <- itoken(train_cleaned$interaction, 
                         preprocessor = tolower,
                         tokenizer = cleaning_tokenizer, 
                         progressbar = FALSE)

vocab_interaction <- create_vocabulary(it_interaction, stopwords = stop_words, ngram = c(1L, 2L)) %>%
  prune_vocabulary(term_count_min = 50, doc_proportion_max=0.3)

# vectorizer
vectorizer_interaction <- vocab_vectorizer(vocab_interaction)

# Convert the training documents into a DTM
dtm_interaction <- create_dtm(it_interaction, vectorizer_transit)

# split tr/va
set.seed(1)
tr_rows <- sample(nrow(train), 0.7*nrow(train))
tr_dtm_interaction <- dtm_interaction[tr_rows,]
va_dtm_interaction <- dtm_interaction[-tr_rows,]

# Get the y values
tr_y <- train[tr_rows,]$high_booking_rate
va_y <- train[-tr_rows,]$high_booking_rate

# try ridge model
ridge_model_interaction <- glmnet(tr_dtm_interaction, tr_y, family="binomial", alpha=0, lambda=.01)
pred_ridge_interaction <- predict(ridge_model_interaction, newx = va_dtm_interaction, type="response")
class_ridge_interaction <- ifelse(pred_ridge_interaction > 0.5, "YES", "NO")
acc_ridge_interaction <- mean(ifelse(class_ridge_interaction == va_y, 1,0))
acc_ridge_interaction

# Make a variable importance plot
vip(ridge_model_interaction, num_features = 20)

# create a new variable: interaction_score
# apply the "count_sentiments" function to each interaction
train_interaction <- train_cleaned$interaction
tokenized_interaction <- lapply(strsplit(tolower(as.character(train_interaction)), "\\s+"), function(tokens) {
  tokens <- tokens[tokens != ""]
  return(tokens)
})
sentiment_counts_interaction <- lapply(tokenized_interaction, count_sentiments)
# convert the list of data frames to a single data frame
sentiment_df_interaction <- do.call(rbind, sentiment_counts_interaction) %>%
  mutate(po_ne_diff = positive_count - negative_count)

# add the new variables from text mining into train_cleaned
train_cleaned <- cbind(train_cleaned, interaction_diff = sentiment_df_interaction$po_ne_diff) 
train_cleaned <- train_cleaned %>%
  group_by(market) %>%
  mutate(max_interaction_score = max(interaction_diff),
         interaction_score = interaction_diff/max_interaction_score) %>%
  ungroup()

# replace NA in interaction_score with 0
train_cleaned <- train_cleaned %>%
  replace_na(list(interaction_score = 0))

#===========================4-5. Text mining: training data - space===========================
# EDA & new variable: "space_score" ---------------------
# itoken "space"
it_space <- itoken(train_cleaned$space, preprocessor = tolower, tokenizer = cleaning_tokenizer, progressbar = FALSE)

# create vocabulary
vocab_space <- create_vocabulary(it_space, stopwords = stop_words, ngram = c(1L, 2L)) %>%
  prune_vocabulary(term_count_min = 50, doc_proportion_max = 0.3)

# vectorizer
vectorizer_space <- vocab_vectorizer(vocab_space)

# Convert the training documents into a DTM and make it a binary BOW matrix
dtm_space <- create_dtm(it_space, vectorizer_space)

# Split into training and validation sets
set.seed(1)
tr_rows <- sample(nrow(train_cleaned), 0.7 * nrow(train_cleaned))
tr_dtm_space <- dtm_space[tr_rows, ]
va_dtm_space <- dtm_space[-tr_rows, ]

# Get the y values
tr_y <- train_cleaned[tr_rows, ]$high_booking_rate
va_y <- train_cleaned[-tr_rows, ]$high_booking_rate

# build ridge model
ridge_model_space <- glmnet(tr_dtm_space, tr_y, family = "binomial", alpha = 0, lambda = 0.01)
pred_ridge_space <- predict(ridge_model_space, newx = va_dtm_space, type = "response")
class_ridge_space <- ifelse(pred_ridge_space > 0.5, "YES", "NO")
acc_ridge_space <- mean(ifelse(class_ridge_space == va_y, 1, 0))
acc_ridge_space

# Variable importance plot
vip(ridge_model_space, num_features = 20)

# create a new variable: interaction_score
# Apply the function to each space
train_space <- train_cleaned$space
tokenized_space <- lapply(strsplit(tolower(as.character(train_space)), "\\s+"), function(tokens) {
  # Remove empty tokens
  tokens <- tokens[tokens != ""]
  return(tokens)
})
sentiment_counts_space <- lapply(tokenized_space, count_sentiments)
sentiment_df_space <- do.call(rbind, sentiment_counts_space) %>%
  mutate(po_ne_diff = positive_count - negative_count)

# add the new variables from text mining into train_cleaned
train_cleaned <- cbind(train_cleaned, space_diff = sentiment_df_space$po_ne_diff)
train_cleaned <- train_cleaned %>%
  group_by(market) %>%
  mutate(max_space_score = max(space_diff),
         space_score = space_diff / max_space_score) %>%
  ungroup()

#===========================4-6. Text mining: test data===========================
# create positive_score, transit_score, interation_score and space_score for test set
# create positive_score from "summary"
# itoken and convert it into dtm
it_summary_test <- itoken(test_clean$summary, 
                          preprocessor = tolower, #preprocessing by converting to lowercase
                          tokenizer = cleaning_tokenizer, 
                          progressbar = FALSE)
dtm_test_clean <- create_dtm(it_summary_test, vectorizer_summary)

summary_test <- test_clean$summary
tokenized_test <- lapply(strsplit(tolower(as.character(summary_test)), "\\s+"), function(tokens) {
  # Remove empty tokens
  tokens <- tokens[tokens != ""]
  return(tokens)
})

# Define positive and negative words
positive_words <- bing_positive$word
negative_words <- bing_negative$word

# Apply the "count_sentiments" function to each summary
sentiment_counts_test <- lapply(tokenized_test, count_sentiments)

# Convert the list of data frames to a single data frame
sentiment_df_test <- do.call(rbind, sentiment_counts_test) %>%
  mutate(po_ne_diff = positive_count - negative_count,
         max_positive_score = max(po_ne_diff),
         positive_score = po_ne_diff/max_positive_score)

# add "positive_score" from text mining into test_clean
test_clean <- cbind(test_clean, positive_score = sentiment_df_test$positive_score)

# create transit_score from "transit"
it_transit_test <- itoken(test_clean$transit, 
                          preprocessor = tolower,
                          tokenizer = cleaning_tokenizer, 
                          progressbar = FALSE)

dtm_test_clean_transit <- create_dtm(it_transit_test, vectorizer_transit)

transit_test <- test_clean$transit
tokenized_test_transit <- lapply(strsplit(tolower(as.character(transit_test)), "\\s+"), function(tokens) {
  # Remove empty tokens
  tokens <- tokens[tokens != ""]
  return(tokens)
})

# Apply the "count_sentiments" function to each summary
sentiment_counts_test_transit <- lapply(tokenized_test_transit, count_sentiments)

# Convert the list of data frames to a single data frame
sentiment_df_test_transit <- do.call(rbind, sentiment_counts_test_transit) %>%
  mutate(po_ne_diff = positive_count - negative_count,
         max_positive_score = max(po_ne_diff),
         transit_score = po_ne_diff/max_positive_score)

# add "transit_score" from text mining into test_clean
test_clean <- cbind(test_clean, transit_score = sentiment_df_test_transit$transit_score)

# create interaction_score from "interaction"
it_interaction_test <- itoken(test_clean$interaction, 
                              preprocessor = tolower,
                              tokenizer = cleaning_tokenizer, 
                              progressbar = FALSE)

dtm_test_clean_interaction <- create_dtm(it_interaction_test, vectorizer_interaction)

interaction_test <- test_clean$interaction
tokenized_test_interaction <- lapply(strsplit(tolower(as.character(interaction_test)), "\\s+"), function(tokens) {
  # Remove empty tokens
  tokens <- tokens[tokens != ""]
  return(tokens)
})


# Apply the "count_sentiments" function to each interaction
sentiment_counts_test_interaction <- lapply(tokenized_test_interaction, count_sentiments)

# Convert the list of data frames to a single data frame
sentiment_df_test_interaction <- do.call(rbind, sentiment_counts_test_interaction) %>%
  mutate(po_ne_diff = positive_count - negative_count,
         max_positive_score = max(po_ne_diff),
         interaction_score = po_ne_diff/max_positive_score)

# add "interaction_score" from text mining into test_clean
test_clean <- cbind(test_clean, interaction_score = sentiment_df_test_interaction$interaction_score)

test_clean <- test_clean %>%
  replace_na(list(interaction_score = 0))

# create space_score from "space"
it_space_test <- itoken(test_clean$space, 
                        preprocessor = tolower,
                        tokenizer = cleaning_tokenizer, 
                        progressbar = FALSE)
dtm_test_clean_space <- create_dtm(it_space_test, vectorizer_space)

space_test <- test_clean$space
tokenized_test_space <- lapply(strsplit(tolower(as.character(space_test)), "\\s+"), function(tokens) {
  # Remove empty tokens
  tokens <- tokens[tokens != ""]
  return(tokens)
})


# Apply the "count_sentiments" function to each space
sentiment_counts_test_space <- lapply(tokenized_test_space, count_sentiments)

# Convert the list of data frames to a single data frame
sentiment_df_test_space <- do.call(rbind, sentiment_counts_test_space) %>%
  mutate(po_ne_diff = positive_count - negative_count,
         max_positive_score = max(po_ne_diff),
         space_score = po_ne_diff/max_positive_score)

# add "space_score" from text mining into test_clean
test_clean <- cbind(test_clean, space_score = sentiment_df_test_space$space_score)

#===========================5. EDA on various features===========================
#===========================5-1. EDA: Amenities clustering===========================
# Find the best k for k-means
# possible_centers <- c(1:30)
# num_centers <- length(possible_centers)
# cluster_cohesion <- rep(0, num_centers)
# for (i in c(1:num_centers)){
# centers <- possible_centers[i]
# km.out_amenities <- kmeans(amenities_matrix, centers=centers, nstart=10)
# total_within_distance <- sum(km.out_amenities$withinss)
# cluster_cohesion[i] <- total_within_distance} 
# best k = 10

# Retrain the model using k=10
km.out_amenities <- kmeans(amenities_matrix, centers=10, nstart=10)
amenities_assignments <- as.factor(km.out_amenities$cluster)

# Visualization
# since availability_rate_90 is also represents the popularity of the property
# so here we treated it as y-axis
train_cleaned_2 <- cbind(train_cleaned, amenities_assignments)
train_cleaned_2 %>%
  ggplot(aes(market, availability_rate_90, col = amenities_assignments)) +
  geom_point() + 
  theme_bw() + 
  theme(panel.border = element_blank(), 
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), 
        axis.line = element_line(colour = "black"),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  ggtitle("Amenities cluster v.s. availability_rate_90 in different market")

centers <- as.data.frame(t(km.out_amenities$centers)) %>%
  rename(Center1 = 1, Center2 = 2, Center3 = 3, Center4 = 4,
         Center5 = 5, Center6 = 6, Center7 = 7, Center8 = 8,
         Center9 = 9, Center10 = 10)

cluster_target_rate <- train_cleaned_2 %>%
  group_by(amenities_assignments) %>%
  summarise(target_rate = mean(ifelse(high_booking_rate == "YES", 1, 0))) %>%
  arrange(desc(target_rate))
cluster_target_rate
# cluster 7 has the highest proportion of high_booking_rate

cluster7_max <- centers %>%
  select(Center7) %>%
  filter(Center7 > 0.6) %>%
  arrange(desc(Center7))
cluster7_max

cluster3_max <- centers %>%
  select(Center3) %>%
  filter(Center3 > 0.6) %>%
  arrange(desc(Center3))
cluster3_max

cluster2_max <- centers %>%
  select(Center2) %>%
  filter(Center2 > 0.6) %>%
  arrange(desc(Center2))
cluster2_max

cluster1_max <- centers %>%
  select(Center1) %>%
  filter(Center1 > 0.6) %>%
  arrange(desc(Center1))
cluster1_max

cluster6_max <- centers %>%
  select(Center6) %>%
  filter(Center6 > 0.6) %>%
  arrange(desc(Center6))
cluster6_max

cluster9_max <- centers %>%
  select(Center9) %>%
  filter(Center9 > 0.6) %>%
  arrange(desc(Center9))
cluster9_max

cluster5_max <- centers %>%
  select(Center5) %>%
  filter(Center5 > 0.6) %>%
  arrange(desc(Center5))
cluster5_max

cluster8_max <- centers %>%
  select(Center8) %>%
  filter(Center8 > 0.6) %>%
  arrange(desc(Center8))
cluster8_max

cluster4_max <- centers %>%
  select(Center4) %>%
  filter(Center4 > 0.6) %>%
  arrange(desc(Center4))
cluster4_max

cluster10_max <- centers %>%
  select(Center10) %>%
  filter(Center10 > 0.6) %>%
  arrange(desc(Center10))
cluster10_max

#===========================5-2. EDA: hosting_year ===========================

averages <- aggregate(train_cleaned$availability_rate_90 ~ train_cleaned$host_year, FUN = mean)
averages <- averages[complete.cases(averages), ]

ggplot(averages, aes(x=`train_cleaned$host_year`, y=`train_cleaned$availability_rate_90`)) +
  geom_point() +
  geom_smooth(method=lm, se=FALSE, color="red") +
  labs(x = "Total host year", 
       y = "Positive score", 
       title = "Relationship between host year and availability")

#===========================5-3. EDA: positive score v.s price ===========================

averages_2 <- aggregate(train_cleaned$price_per_person ~ train_cleaned$positive_score, FUN = mean)
averages_2 <- averages_2[complete.cases(averages), ]

ggplot(averages_2, aes(x=`train_cleaned$price_per_person`, y=`train_cleaned$positive_score`)) +
  geom_point() +
  geom_smooth(method=lm, se=FALSE, color="red") +
  labs(x = "Price", 
       y = "Positive score", 
       title = "Relationship between positive score and price")

#===========================5-4. EDA: price vs availability ===========================

averages_3 <- aggregate(train_cleaned$price ~ train_cleaned$availability_rate_90, FUN = mean)
averages_3 <- averages_3[complete.cases(averages), ]

ggplot(averages_3, aes(x=`train_cleaned$availability_rate_90`, y=`train_cleaned$price`)) +
  geom_point() +
  geom_smooth(method=lm, se=FALSE, color="red") +
  labs(x = "Listing Availability within 90 days", 
       y = "Price", 
       title = "Relationship between Price and Listing Availability")
#===========================5-5. EDA: high_booking_rate v.s distance to airport===========================
install.packages("vioplot")
library(vioplot)
vioplot(train_cleaned$`distance_to_airport/1000` ~ train_cleaned$high_booking_rate, 
        xlab = "Is High Booking Rate", 
        ylab = "Distance from Airport", 
        main = "Relationship between High Booking Rate and Distance from Airport")

#===========================6. Modeling===========================
#===========================6-1. Split training data into train/valid/small===========================
set.seed(1)
train_inst <- sample(nrow(train_cleaned), 0.7*nrow(train_cleaned))
# training set
training_all <- train_cleaned[train_inst,]
training_X <- training_all %>% select(-high_booking_rate)
training_y <- training_all$high_booking_rate

heldout <- train_cleaned[-train_inst,]
# validation set
valid_inst <- sample(nrow(heldout), .9*nrow(heldout))
valid_all <- heldout[valid_inst,]
valid_X <- valid_all %>% select(-high_booking_rate)
valid_y <- valid_all$high_booking_rate

# small set for final check (to avoid overfitting in validation data)
small_all <- heldout[-valid_inst,]
small_X <- small_all %>% select(-high_booking_rate)
small_y <- small_all$high_booking_rate

#===========================6-2. Feature selection===========================
# save all of the dummy variables from amenities in amenities_var
amenities_var <- colnames(amenities_matrix)
selected_var <- c("accommodates", "bedrooms", "beds", "cancellation_policy", "has_cleaning_fee", 
                   "host_total_listings_count", "price", "ppp_ind", "property_category", "bed_category", 
                   "bathrooms", "charges_for_extra", "host_acceptance", "host_response", "has_min_nights", "market", 
                   "max_nights_group", "has_license", "high_booking_rate","host_year", 
                   "availability_rate_30", "availability_rate_60", "availability_rate_90", "availability_rate_365",
                   "positive_score", "transit_score", "super_host", "has_profile_pic", "identity_verified", 
                  "location_exact", "instant_bookable", "requires_license", "guest_profile_pic_required", 
                  "guest_phone_verification_required", "interaction_score", "has_security_deposit", "space_score","nearby_airport",
                  "nearby_attractions","jurisdiction_category","has_weekly_discount","has_monthly_discount")
# combine the above variables and amenities_var
selected_var <- c(selected_var, amenities_var)

# create data frame for selected var for training and valid respectively
# These two are for logistic or regression data (because it includes "high_booking_rate")
train_selected <- training_all %>% 
  select(all_of(selected_var))

valid_selected <- valid_all %>% 
  select(all_of(selected_var))

# for other models
# split X and y
training_X_selected <- train_selected %>%
  select(-high_booking_rate)

valid_X_selected <- valid_selected %>%
  select(-high_booking_rate)

#===========================6-3. Try multiple models===========================
#===========================6-3-1. Logistic model===========================
# dummy variables
# prepare for training data
dummy_train_var <- dummyVars(~ ., data = train_selected, fullRank = TRUE)
dummy_train <- data.frame(predict(dummy_train_var, newdata = train_selected))

# prepare for valid data
dummy_valid_var <- dummyVars(~ ., data = valid_selected, fullRank = TRUE)
dummy_valid <- data.frame(predict(dummy_valid_var, newdata = valid_selected))

# logistic model
log_model <- glm(dummy_train$high_booking_rate.YES~., data = dummy_train, family = "binomial")
log_model

# Ppredict the probabilities for the test data
prob_log <- predict(log_model, newdata = dummy_valid, type = "response")
prob_log

# check the accuracy
pred_log <- ifelse(prob_log > .5, "YES", "NO")
acc_log <- mean(ifelse(pred_log == valid_y, 1, 0))
acc_log

# create a ROC curve object
roc_log <- prediction(prob_log, valid_y)

# calculate the AUC
auc_score_log <- performance(roc_log, "auc")@y.values[[1]]

# Plot the ROC curve
roc_curve_log <- performance(roc_log, "tpr", "fpr")
plot(roc_curve_log, col = "blue", main = "ROC Curve", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "red")  # Diagonal reference line
legend("bottomright", legend = paste("AUC =", round(auc_score_log, 4)), col = "blue", lwd = 2)

#===========================6-3-2. Random Forest model===========================
# random forest model
rf.mod <- randomForest(x=training_X_selected,
                       y=training_y,
                       mtry=12, ntree=1000,
                       importance=TRUE)

prob_rf <- predict(rf.mod, newdata=valid_selected, type = "prob")
prob_rf_yes <- prob_rf[,"YES"]

pred_rf_binary <- ifelse(prob_rf_yes > 0.5, "YES", "NO")
acc_rf <- mean(ifelse(pred_rf_binary == valid_y, 1, 0))
acc_rf

# importance(rf.mod)
varImpPlot(rf.mod)

# Create a ROC curve object
roc_rf <- prediction(prob_rf_yes, valid_y)

# Calculate the AUC
auc_score_rf <- performance(roc_rf, "auc")@y.values[[1]]

# Plot the ROC curve
roc_curve_rf <- performance(roc_rf, "tpr", "fpr")
plot(roc_curve_rf, col = "blue", main = "ROC Curve", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "red")  # Diagonal reference line
legend("bottomright", legend = paste("AUC =", round(auc_score_rf, 4)), col = "blue", lwd = 2)

#=================================6-3-3. Ridge Model====================================
# data preparation
dummy_train_2 <- dummy_train %>% select(-high_booking_rate.YES)
dummy_train_y <- dummy_train$high_booking_rate.YES
dummy_valid_2 <- dummy_valid %>% select(-high_booking_rate.YES)
dummy_valid_y <- dummy_valid$high_booking_rate.YES

# set up the grid of lambda values
lambda_values <- 10^seq(-7, 7, length.out = 100)

# fit the Ridge-penalized logistic regression model using cross-validation
cv_fit_ridge <- cv.glmnet(as.matrix(dummy_train_2), dummy_train_y, family = "binomial", alpha = 0, lambda = lambda_values, nfolds = 5)

# plot the fitting curve
plot(cv_fit_ridge)

# find the optimal lambda
optimal_lambda <- cv_fit_ridge$lambda.min
cat("Optimal lambda for Ridge model:", optimal_lambda, "\n")

# fit the final Ridge model with the optimal lambda
final_ridge_model <- glmnet(as.matrix(dummy_train_2), dummy_train_y, family = "binomial", alpha = 0, lambda = optimal_lambda)

# make predictions with the best lambda Ridge model on the training data
preds_ridge <- predict(final_ridge_model, newx = as.matrix(dummy_valid_2), type = "response")

# plot ROC
roc_ridge <- prediction(preds_ridge, dummy_valid_y)
# create a performance object
roc_curve_ridge <- performance(roc_ridge, "tpr", "fpr")
# calculate AUC
auc_ridge <- performance(roc_ridge, measure = "auc")@y.values[[1]]
# plot the ROC curve
plot(roc_curve_ridge, col = "blue", main = "ROC Curve", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "red")  # Diagonal reference line
legend("bottomright", legend = paste("AUC =", round(auc_ridge, 4)), col = "blue", lwd = 2)

# Convert predicted probabilities to class labels
preds_ridge_labels <- ifelse(preds_ridge > 0.5, 1, 0)

# Calculate accuracy
acc_ridge <- sum(preds_ridge_labels == dummy_valid_y) / length(valid_y)
cat("Accuracy for Ridge model on validation set:", acc_ridge, "\n")

# save the ridge coefficient in coef_ridge
coef_ridge <- coef(cv_fit_ridge)

#=================================6-3-4. Lasso Model====================================
# Set up the grid of lambda values
lambda_values <- 10^seq(-7, 7, length.out = 100)

# Fit the Ridge-penalized logistic regression model using cross-validation
cv_fit_lasso <- cv.glmnet(as.matrix(dummy_train_2), dummy_train_y, family = "binomial", alpha = 1, lambda = lambda_values, nfolds = 5)

# Plot the fitting curve
plot(cv_fit_lasso)

# Find the optimal lambda
optimal_lambda_lasso <- cv_fit_lasso$lambda.min
cat("Optimal lambda for Lasso model:", optimal_lambda_lasso, "\n")

# Fit the final Lasso model with the optimal lambda
final_lasso_model <- glmnet(as.matrix(dummy_train_2), dummy_train_y, family = "binomial", alpha = 1, lambda = optimal_lambda_lasso)

# Make predictions with the best lambda Ridge model on the training data
preds_lasso <- predict(final_lasso_model, newx = as.matrix(dummy_valid_2), type = "response")

# Plot ROC
roc_lasso <- prediction(preds_lasso, dummy_valid_y)
# Create a performance object
roc_curve_lasso <- performance(roc_lasso, "tpr", "fpr")
# Calculate AUC
auc_lasso <- performance(roc_lasso, measure = "auc")@y.values[[1]]
# Plot the ROC curve
plot(roc_curve_lasso, col = "blue", main = "ROC Curve", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "red")  # Diagonal reference line
legend("bottomright", legend = paste("AUC =", round(auc_lasso, 4)), col = "blue", lwd = 2)

# Calculate the AUC
print(paste("AUC:", round(auc_lasso, 4)))

# Convert predicted probabilities to class labels
preds_lasso_labels <- ifelse(preds_lasso > 0.5, 'YES', 'NO')
# Calculate accuracy
acc_lasso<- sum(preds_lasso_labels == valid_y) / length(valid_y)
cat("Accuracy for Lasso model on validation set:", acc_lasso, "\n")

# variable selection
# coefficient
coef_lasso <- coef(cv_fit_lasso)
coef_matrix <- as.matrix(coef_lasso)

# select the variables coefficient not equals to 0
coef_df <- as.data.frame(coef_matrix)
coef_df_filter <- coef_df %>%
    filter(s1 != 0)
# for exploration, see which variables' coef = 0
coef_df_0 <- coef_df %>%
  filter(s1 == 0)

coef_df_filter

#===========================6-3-4-2. Feature selection based on lasso model===========================
# remove the features which were shrinked to 0 by lasso model
selected_var2 <- rownames(coef_df_filter)
selected_var2 <- selected_var2[selected_var2 != "(Intercept)"]
target_var <- "high_booking_rate.YES"
selected_var2 <- c(selected_var2, target_var)
selected_var2

# define training and valid data set using select variables
dummy_train_new <- dummy_train %>% 
  select(all_of(selected_var2))

dummy_valid_new <- dummy_valid %>% 
  select(all_of(selected_var2))

#================================6-3-5. Xgboost Model======================================
# data preparation
dummy_train_matrix<- as.matrix(dummy_train_new %>% select(-high_booking_rate.YES))
dummy_valid_matrix <- as.matrix(dummy_valid_new %>% select(-high_booking_rate.YES))

# train the model
xgb.mod <- xgboost(data = dummy_train_matrix,
               label = dummy_train_y,
               max.depth = 10,
               eta = 1,
               nrounds = 2500,
               objective = "binary:logistic")

# predict using xgb.mod
preds_xgb <- predict(xgb.mod, newdata = dummy_valid_matrix)
preds_xgb_pred <- prediction(preds_xgb, dummy_valid_y)

# calculate AUC
auc_xgb <- performance(preds_xgb_pred, measure = "auc")@y.values[[1]]
cat("AUC for XGBoost model on validation set:", auc_xgb, "\n")
xgb_classifications <- ifelse(preds_xgb > 0.5, 1, 0)
xgb_accuracy <- mean(ifelse(xgb_classifications == dummy_valid_y, 1, 0))
paste('Accuracy of the boosting based method is:', xgb_accuracy)

# plot the ROC curve
roc_curve_xgb <- performance(preds_xgb_pred, "tpr", "fpr")
plot(roc_curve_xgb, col = "blue", main = "ROC Curve", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "red")  # Diagonal reference line
legend("bottomright", legend = paste("AUC =", round(auc_xgb, 4)), col = "blue", lwd = 2)

#================================6-3-6. Boosting Model======================================
# train the model
boost.mod <- gbm(high_booking_rate.YES~.,data=dummy_train_new,
                 distribution="bernoulli",
                 n.trees=2000,
                 interaction.depth=4)
# predict using boost.mod
boost_preds <- predict(boost.mod,
                       newdata=dummy_valid_new,
                       type='response',
                       n.trees=2000)
# create the predicted label using cutoff = 0.5
boost_class <- ifelse(boost_preds > 0.5, 1, 0)
boost_acc <- mean(ifelse(boost_class == dummy_valid_new$high_booking_rate.YES, 1, 0))

# create a ROC curve object
roc_boost <- prediction(boost_preds, dummy_valid$high_booking_rate.YES)

# calculate the AUC
auc_score_boost <- performance(roc_boost, "auc")@y.values[[1]]

# plot the ROC curve
roc_curve_boost <- performance(roc_boost, "tpr", "fpr")
plot(roc_curve_boost, col = "blue", main = "ROC Curve", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "red")  # Diagonal reference line
legend("bottomright", legend = paste("AUC =", round(auc_score_boost, 4)), col = "blue", lwd = 2)

#===========================7. Choosing the winning model===========================
#===========================7-1. Review the model performance of all models===========================
# Plot the roc curve from all models
# These roc and auc are based on validation data
plot(roc_curve_log, col = "red", lwd = 2)
plot(roc_curve_ridge, add=T, col = "blue", ylab = '', lwd = 2)
plot(roc_curve_lasso, add=T, col = "purple", ylab = '', lwd = 2)
plot(roc_curve_rf, add=T, col = "grey", ylab = '', lwd = 2)
plot(roc_curve_bst, add=T, col = "orange", ylab = '', lwd = 2)
plot(roc_curve_boost, add=T, col = "green", ylab = '', lwd = 2)
legend("bottomright", legend = c("Logistic", "Ridge", "Lasso", "Random forest","Xgboost","Boosting"),
       col = c("red", "blue", "purple", "grey","orange", "green"), lty = 1)
auc_score_log
auc_lasso
auc_ridge
auc_score_rf
auc_bst
auc_score_boost

#===========================7-2. Cross validation===========================
# some variables are not in test so not excluded
not_selected_var <- c("roll.in.shower.with.shower.bench.or.chair", "waterfront", 
                      "grab.rails.for.shower.and.toilet","tub.with.shower.bench",
                      "disabled.parking.spot", "wide.clearance.to.shower.and.toilet", 
                      "cleaning.before.checkout", "wide.hallway.clearance")
selected_var_final <- selected_var[!selected_var %in% not_selected_var]
# create a data frame having the final selected variables 
train_whole_selected <- train_cleaned %>% select(all_of(selected_var_final))

# dummy the all dataset
dummy_train_all_var <- dummyVars(~ ., data = train_whole_selected, fullRank = TRUE)
dummy_train_all <- data.frame(predict(dummy_train_all_var, newdata = train_whole_selected))

# split X and y
training_X_whole_selected <- train_whole_selected
training_y_whole <- train_cleaned$high_booking_rate

# randomly shuffle the training data
trained_shuffle <- dummy_train_all[sample(nrow(dummy_train_all)),]
# create trained_shuffle_2 for random forest model
trained_shuffle_2 <- training_X_whole_selected[sample(nrow(training_X_whole_selected)),]
# define k = the number of folds
k <- 10

# separate data into k equally-sized folds
folds <- cut(seq(1,nrow(trained_shuffle)),breaks=k,labels=FALSE)

# Make a vectors of zeros to store model performance in each fold
# auc
log_auc_cv <- rep(0, k)
ridge_auc_cv <- rep(0, k)
lasso_auc_cv <- rep(0, k)
rf_auc_cv <- rep(0, k)
xgb_auc_cv <- rep(0, k)
boost_auc_cv <- rep(0, k)

# cross-validation for logistic, ridge, lasso, xgboost, boosting model
for(i in 1:k) {
  valid_inds <- which(folds == i)
  valid_fold <- trained_shuffle[valid_inds, ]
  train_fold <- trained_shuffle[-valid_inds, ]
  train_actuals <- train_fold$high_booking_rate.YES
  valid_actuals <- valid_fold$high_booking_rate
  
  # convert data frames to matrices (for ridge and lasso)
  train_fold_x <- train_fold %>% select(-high_booking_rate.YES)
  valid_fold_x <- valid_fold %>% select(-high_booking_rate.YES)
  train_fold_x_matrix <- as.matrix(train_fold_x)
  valid_fold_x_matrix <- as.matrix(valid_fold_x)

  # train models
  log_model_cv <- glm(high_booking_rate.YES~., data = train_fold, family = "binomial")
  ridge_model_cv <- glmnet(train_fold_x_matrix, train_actuals, alpha = 0, lambda = optimal_lambda, family = "binomial")
  lasso_model_cv <- glmnet(train_fold_x_matrix, train_actuals, alpha = 1, lambda = optimal_lambda_lasso, family = "binomial")
  xgb.mod_cv <- xgboost(data = train_fold_x_matrix,
                        label = train_actuals,
                        max.depth = 10,
                        eta = 1,
                        nrounds = 2500,
                        objective = "binary:logistic")
  boost.mod_cv <- gbm(high_booking_rate.YES ~ ., data = train_fold,
                      distribution = "bernoulli",
                      n.trees = 1000,
                      interaction.depth = 4)

  # predict probabilities
  prob_log_cv <- predict(log_model_cv, newdata = valid_fold, type = "response")
  probs_ridge_cv <- predict(ridge_model_cv, newx = valid_fold_x_matrix, type = "response")
  probs_lasso_cv <- predict(lasso_model_cv, newx = valid_fold_x_matrix, type = "response")
  probs_xgb_cv <- predict(xgb.mod_cv, newdata = valid_fold_x_matrix)
  boost_probs_cv <- predict(boost.mod_cv, newdata = valid_fold, type = 'response', n.trees = 1000)
  
  # create ROC objects
  roc_log_cv <- prediction(prob_log_cv, valid_actuals)
  roc_ridge_cv <- prediction(probs_ridge_cv, valid_actuals)
  roc_lasso_cv <- prediction(probs_lasso_cv, valid_actuals)
  roc_xg_cv <- prediction(probs_xgb_cv, valid_actuals)
  
  # extract AUC values
  log_auc_cv[i] <- performance(roc_log_cv, "auc")@y.values[[1]]
  ridge_auc_cv[i] <- performance(roc_ridge_cv, "auc")@y.values[[1]]
  lasso_auc_cv[i] <- performance(roc_lasso_cv, "auc")@y.values[[1]]
  xgb_auc_cv[i] <- performance(roc_xg_cv, measure = "auc")@y.values[[1]]
  boost_auc_cv[i] <- performance(prediction(boost_probs_cv, valid_actuals), "auc")@y.values[[1]]
}

# cross-validation for random forest model
for(i in 1:k) {
  valid_inds <- which(folds == i)
  valid_fold_2 <- trained_shuffle_2[valid_inds, ]
  train_fold_2 <- trained_shuffle_2[-valid_inds, ]
  train_actuals_2 <- train_fold_2$high_booking_rate
  valid_actuals_2 <- valid_fold_2$high_booking_rate
  
  # train Random Forest model
  rf.mod_cv <- randomForest(high_booking_rate ~ .,
                            data = train_fold_2,
                            mtry = 12,
                            ntree = 1000,
                            importance = TRUE)
  
  # predict probabilities for the validation data
  prob_rf_cv <- predict(rf.mod_cv, newdata = valid_fold_2, type = "prob")
  
  # extract probabilities for class YES
  prob_rf_yes_cv <- prob_rf_cv[, "YES"]
  
  # create ROC curve object
  roc_rf_cv <- prediction(prob_rf_yes_cv, valid_actuals_2)
  
  # calculate AUC and store in rf_auc_cv
  rf_auc_cv[i] <- performance(roc_rf_cv, "auc")@y.values[[1]]
}
#===========================7-3. Double check by testing on small held-out data===========================
# Test the top 3 model performance on held-out data: ROC & AUC
# Review: what is the small data?
# In "6-1. Split training data into train/valid/small", we create a small set from training data set.
# The purpose is to use it double check the auc of the final models to ensure it is not over-fitting to validation set
# 1. random forest on held-out data
small_X <- small_all %>% 
  select(all_of(selected_var))%>%
  select(-high_booking_rate)
small_y <- small_all$high_booking_rate

prob_rf_held <- predict(rf.mod, newdata=small_X, type = "prob")
prob_rf_yes_held <- prob_rf_held[,"YES"]
pred_rf_binary_held <- ifelse(prob_rf_yes_held > 0.5, "YES", "NO")
acc_rf_held <- mean(ifelse(pred_rf_binary_held == small_y, 1, 0))
acc_rf_held

roc_rf_held <- prediction(prob_rf_yes_held, small_y)

# Calculate the AUC
auc_score_rf <- performance(roc_rf_held, "auc")@y.values[[1]]
auc_score_rf

# 2. xgboost on held-out data
small_X <-small_all %>% 
  select(all_of(selected_var))
dummy_small_var <- dummyVars(~ ., data = small_X, fullRank = TRUE)
dummy_small <- data.frame(predict(dummy_small_var, newdata = small_X))

dummy_small <- dummy_small %>% 
  select(all_of(selected_var2))

dummy_small_x<- dummy_small%>%
  select(-high_booking_rate.YES)

dummy_small_y<-dummy_small$high_booking_rate.YES

dummy_small_x_matrix<- as.matrix(dummy_small_x)
preds_bst_held <- predict(bst, newdata = dummy_small_x_matrix)

preds_bst_pred__held <- prediction(preds_bst_held, dummy_small_y)
auc_bst_held <- performance(preds_bst_pred__held, measure = "auc")@y.values[[1]]
auc_bst_held
bst_classifications_held <- ifelse(preds_bst_held>0.5, 1, 0)
bst_accuracy_held <- mean(ifelse(bst_classifications_held == dummy_small_y, 1, 0))
paste('Accuracy of the boosting based method is:', bst_accuracy_held)

# 3. boosting on held-out data
dummy_small_x<- dummy_small
boost_preds_held <- predict(boost.mod,
                            newdata=dummy_small_x,
                            type='response',
                            n.trees=1000)

boost_class_held <- ifelse(boost_preds_held>0.5, 1, 0)
boost_acc_held <- mean(ifelse(boost_class_held==dummy_small_x$high_booking_rate.YES,1,0))

# Create a ROC curve object
roc_boost_held <- prediction(boost_preds_held, dummy_small_x$high_booking_rate.YES)

# Calculate the AUC
auc_score_boost <- performance(roc_boost_held, "auc")@y.values[[1]]
auc_score_boost

#===========================8. Re-train the winning model using the whole training set===========================
# data preparation
# whole training set
# some variables are not in test so not excluded
not_selected_var <- c("high_booking_rate", 
                      "roll.in.shower.with.shower.bench.or.chair", "waterfront", 
                      "grab.rails.for.shower.and.toilet","tub.with.shower.bench",
                      "disabled.parking.spot", "wide.clearance.to.shower.and.toilet", 
                      "cleaning.before.checkout", "wide.hallway.clearance")
selected_var_final <- selected_var[!selected_var %in% not_selected_var]

train_whole_selected <- train_cleaned %>% 
  select(all_of(selected_var_final))

# split X and y
training_X_whole_selected <- train_whole_selected
training_y_whole <- train_cleaned$high_booking_rate

# unlabeled test set
test_selected <- test_clean %>% select(all_of(selected_var_final))

#===========================9-1. predict the test data using the top 3 models===========================  
# 1. training random forest model
rf.mod_final <- randomForest(x=training_X_whole_selected,
                       y=training_y_whole,
                       mtry=12, ntree=1000,
                       importance=TRUE)
# predict the the probability of test data's target variable
prob_rf <- predict(rf.mod_final, newdata = test_selected, type = "prob")
prob_rf_yes <- prob_rf[,"YES"]
prob_rf_yes

# 2. training xgboost model
not_selected_var <- c("roll.in.shower.with.shower.bench.or.chair", "waterfront", 
                      "grab.rails.for.shower.and.toilet","tub.with.shower.bench",
                      "disabled.parking.spot", "wide.clearance.to.shower.and.toilet", 
                      "cleaning.before.checkout", "wide.hallway.clearance")
selected_var_final <- selected_var[!selected_var %in% not_selected_var]

test_selected <- test_selected%>%
  mutate(has_min_nights=as.factor(has_min_nights),
         has_weekly_discount = as.factor(has_weekly_discount),
         has_monthly_discount = as.factor(has_monthly_discount))


# split X and y
training_X_whole_selected <- train_whole_selected
training_y_whole <- train_cleaned$high_booking_rate


dummy_whole_var <- dummyVars(~ ., data = training_X_whole_selected, fullRank = TRUE)
dummy_whole <- data.frame(predict(dummy_whole_var, newdata = training_X_whole_selected))

dummy_whole_x<- dummy_whole%>%
  select(-high_booking_rate.YES)
dummy_whole_y<-dummy_whole$high_booking_rate.YES
dummy_whole_matrix<- as.matrix(dummy_whole_x)
bst <- xgboost(data = dummy_whole_matrix,
               label = dummy_whole_y,
               max.depth = 10,
               eta = 1,
               nrounds = 2500,
               objective = "binary:logistic")

dummy_test_var<- dummyVars(~ ., data = test_selected, fullRank = TRUE)
dummy_test <- data.frame(predict(dummy_test_var, newdata = test_selected))
dummy_test_matrix<- as.matrix(dummy_test)
colnames(dummy_test_matrix) <- names(dummy_test)
preds_bst <- predict(bst, newdata = dummy_test_matrix)

#3. training boosting model
boost.mod <- gbm(high_booking_rate.YES~.,data=dummy_whole,
                 distribution="bernoulli",
                 n.trees=1000,
                 interaction.depth=4)
boost_preds <- predict(boost.mod,
                       newdata=dummy_test,
                       type='response',
                       n.trees=1000)

#===========================9-2. The final prediction probability===========================
# calculate the average of the top 3 model as the final submit
if (all(dim(boost_preds) == dim(prob_rf_yes)) && all(dim(prob_rf_yes) == dim(preds_bst))) {
  average_preds <- ((boost_preds + prob_rf_yes + preds_bst) / 3)
  print(average_preds)
} else {
  print("Error: Dimensions of prediction matrices do not match.")
}

# output the probability
write.table(average_preds, "high_booking_rate_group18.csv", row.names = FALSE)

# double check if output probs have NA
check <- read.csv("high_booking_rate_group18.csv")
sum(is.na(check$x))
