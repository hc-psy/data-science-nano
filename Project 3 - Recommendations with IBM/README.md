# Recommendations With IBM

This project aims at constructing a recommendation system to recommend articles to readers, employing ML matrix factorization techniques.

## Overview

### Exploratory Data Analysis

Before making recommendations of any kind, we will need to explore the data we are working with for the project.

1. What is the distribution of how many articles a user interacts with in the dataset?
2. The number of unique articles that have an interaction with a user.
3. The number of unique articles in the dataset (whether they have any interactions or not).
4. The number of unique users in the dataset. (excluding null values)
5. The number of user-article interactions in the dataset.
6. The most viewed article_id

### Rank Based Recommendations

* Functions to return the top articles with most interactions
* Function to return the top articles IDs

### User-User Based Collaborative Filtering

* Re-formatted df to show unique users and articles
* If a user has interacted with an article at least once, the respective user-item pair is marked as 1 or marked as 0 if otherwise.
* Function to find similar users based on similarity of their interactions
* Function to return article names based on their IDs
* Function to return all articles the user interacted
* User-user collaborative Filtering function and it's improved version
* Function that sorts articles by popularity
* Function that sorts users by the number of their activity

### Matrix Factorization

* SVD function and predictions
