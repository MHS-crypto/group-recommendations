# Movie Predictions with Group Recommendations

## Overview

This repository contains a Python file with three methods for predicting movie recommendations within a group, using the MovieLens 100k dataset. The system employs the Pearson correlation method to identify the top 10 users closest to a given user and predicts the top 10 movies for that group.

### Methods

1. **Average Method**
   - For common movies within the group, calculate the average rating based on individual ratings.
   - Return the top 10 rated movies for the group.

2. **Least Misery Method**
   - For common movies within the group, set the group rating as the least rating given by any group member.
   - Return the top 10 rated movies for the group.

3. **Weighted Aggregation Method**
   - Prioritize individuals with similar movie tastes within the group.
   - Assess the alignment of someone's preferences with the overall preferences of the group.
   - Users with significantly different preferences are intentionally given less influence.
   - Adjust each person's ratings based on their assigned weights.
   - Provide more tailored movie recommendations by balancing individual preferences with the overall group taste.
   - Return the top movies based on the weighted aggregation.

### Problems Addressed by Weighted Aggregation Method

- **Disagreement in Preferences:**
  - Previous methods didn't consider cases where people didn't agree on what they liked, leading to potential bias in suggestions.
- **Individual Ratings Influence:**
  - Some users' high or low ratings could excessively influence movie suggestions for everyone.

### Weighted Aggregation Method Working

1. **Calculate Disagreement:**
   - Measure the disagreement between users based on their ratings for common movies.

2. **Calculate Weights:**
   - Assign weights based on the inverse of the sum of disagreements.
   
3. **Adjust Ratings:**
   - Multiply weights with individual ratings, sum the results, and return the top movies.

### Example

- Users A, B, and C with their respective ratings.
- Disagreements calculated between users.
- Weights assigned based on disagreements.
- Final ratings adjusted using weights for movie recommendations.

## Usage

1. Clone the repository.
   ```bash
   git clone https://github.com/MHS-crypto/group-recommendations.git
   ```

2. Run the Python script.
   ```bash
   python group_recommendation.py
   ```

3. Explore the top-rated movies for the group using different prediction methods.

## Conclusion

The Weighted Aggregation Method provides more tailored group suggestions by considering the alignment of users' preferences. It balances individual likes with the overall group preferences, making movie recommendations more personalized for everyone.
