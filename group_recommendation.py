"""
Execution Instructions: The code is written in Pyton and requires Python version of 3.9 or above.
Change the path for data files for users and ratings to load the files.

"""

import pandas as pd
import numpy as np




# importing users data
users = pd.read_csv('File_Path\\ml-100k\\u.user', sep="|", names=["user_id", "age", "gender", "occupation", "zip_code"])
users.head(2)

# importing ratings data
ratings = pd.read_csv('File_Path\\ml-100k\\u.data',  sep='\t', names = ['user_id', 'movie_id', 'rating', 'timestamp']
, encoding='latin-1')

ratings.head(2)

# Now we will merge the ratings and users data on user_id
data = pd.merge(users, ratings, on="user_id")


# creating a pivot table to summarize the data making movie_id as column, taking ratings as values 
# user_id as rows
user_item_matrix = data.pivot_table(index="user_id", columns="movie_id", values="rating")
user_item_matrix = user_item_matrix.fillna(0)




def pearson_corelation(selected_user):
    
    
    similarities = []
    for user in user_item_matrix.index:
        
        if user != selected_user:
            
            common_movies = set(user_item_matrix.columns[user_item_matrix.loc[selected_user].notna() & user_item_matrix.loc[user].notna()])
            if len(common_movies) == 0:
                 return 0  # Users have no common rated movies
            user1_ratings = user_item_matrix.loc[selected_user, common_movies]
            user2_ratings = user_item_matrix.loc[user, common_movies]
            correlation = np.corrcoef(user1_ratings, user2_ratings)[0, 1]
            
            if np.isnan(correlation):
                similarity =  0
                similarities.append((user,similarity))
            else:
                similarity = correlation
                similarities.append((user,similarity))
            
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

    return similarities[:10]


def predict_movie_score(selected_user, target_movie_id, similar_users, user_item_matrix):
    # Calculate the mean rating ¯𝒓𝒂 for the selected user
    active_user_ratings = user_item_matrix.loc[selected_user]
    active_user_mean_rating = active_user_ratings.mean()
    
    weighted_sum = 0
    similarity_sum = 0
    
    for similar_user, similarity_score in similar_users:
        if user_item_matrix.at[similar_user, target_movie_id] != 0:
            # Calculate ¯𝒓𝒃 for the similar user 𝒃
            similar_user_ratings = user_item_matrix.loc[similar_user]
            similar_user_mean_rating = similar_user_ratings.mean()
            
            weighted_sum += similarity_score * (user_item_matrix.at[similar_user, target_movie_id] - similar_user_mean_rating)
            
            similarity_sum += abs(similarity_score)
    
    if similarity_sum == 0:
        return active_user_mean_rating
    
    # Calculate the final prediction
    predicted_score = active_user_mean_rating + weighted_sum / similarity_sum
    return predicted_score


def movie_predictor(user_id):
    similar_users = pearson_corelation(user_id)

    unrated_movies = user_item_matrix.columns[user_item_matrix.loc[user_id] == 0]
    predictions = []

    for movie in unrated_movies:
        score = predict_movie_score(user_id, movie, similar_users, user_item_matrix)
        predictions.append((user_id, movie, score))
    
    top_10_movies = sorted(predictions, key=lambda x: x[2], reverse=True)[:10]


    return top_10_movies




# we will choose members in the group as the 2 closest one to user 3 which are 863 and 616 based on pearson corelation results results
# In the first step we will find top 10 movies for each member in the group
# Aggregate the lists of all users to make a single list with common movies


# Method 1: Average Method
# For this method we will find average rating for each movie and than return the top 10 movies
# based on the average ratings for the group
def group_recommendations_average(*user_data_list):
   

    # Combine data for all users
    all_users_data = []
    for user_data in user_data_list:
        all_users_data.extend(user_data)

    # Create a dictionary to store aggregated ratings for each movie
    aggregated_ratings = {}

    # Aggregate ratings for each movie across all users
    for user_id, movie_id, rating in all_users_data:
        if movie_id not in aggregated_ratings:
            aggregated_ratings[movie_id] = []

        aggregated_ratings[movie_id].append(rating)

    # Calculate the average rating for each movie
    average_ratings = {}
    for movie_id, ratings_list in aggregated_ratings.items():
        average_ratings[movie_id] = sum(ratings_list) / len(ratings_list)

    # Identify the top-rated movies based on average ratings
    top_movies = sorted(average_ratings, key=average_ratings.get, reverse=True)[:10]

    return [(movie_id, average_ratings[movie_id]) for movie_id in top_movies]


# Method 2: Least Misery Method

def group_recommendations_least_misery(*user_data_lists):
    

    # Find common movies among all users
    common_movies = set(entry[1] for entry in user_data_lists[0])  

    for user_data in user_data_lists[1:]:
        common_movies &= set(entry[1] for entry in user_data)

    # Filter user data to include only common movies
    filtered_user_data = [
        [entry for entry in user_data if entry[1] in common_movies]
        for user_data in user_data_lists
    ]

    # Combine data for all users
    all_users_data = []
    for user_data in filtered_user_data:
        all_users_data.extend(user_data)

    # Create a dictionary to store minimum ratings for each movie
    min_ratings = {}

    # Find the minimum rating for each movie across all users
    for user_id, movie_id, rating in all_users_data:
        if movie_id not in min_ratings:
            min_ratings[movie_id] = float('inf')  

        min_ratings[movie_id] = min(min_ratings[movie_id], rating)

    # Identify the top-rated movies based on the Least Misery method
    top_movies = sorted(min_ratings, key=min_ratings.get, reverse=True)[:10]

    # Return the top-rated movies and their minimum ratings
    return [(movie_id, min_ratings[movie_id]) for movie_id in top_movies]


# Method 3: Weighted Aggregation Method


def calculate_disagreement(user_ratings1, user_ratings2):
    squared_diff_sum = sum((rating1 - rating2)**2 for _, _, rating1 in user_ratings1 for _, _, rating2 in user_ratings2)
    return squared_diff_sum**0.5


def weighted_aggregation(*user_data_lists):
     # Step 1: Find common movies among all users
    common_movies = set.intersection(*(set(entry[1] for entry in user_data) for user_data in user_data_lists))

    # Step 2: Filter user data to include only common movies
    flattened_user_data = [
        [entry for entry in user_data if entry[1] in common_movies]
        for user_data in user_data_lists
    ]

    
    filtered_user_data = [flattened_user_data[0][i:i+10] for i in range(0, len(flattened_user_data[0]),10)]
 

    # Step 3: Calculate weights based on disagreements

    
    weights = [1 / (1 + sum(calculate_disagreement(user_data_i, user_data_j)
               for user_data_j in filtered_user_data[i + 1:]))
               for i, user_data_i in enumerate(filtered_user_data)]
    
    print(weights) 

    # Step 4: Combine data for all users, considering weights
    all_users_data = [
        (user_id, movie_id, rating * weights[user_id])
        for user_id, user_data in enumerate(filtered_user_data)
        for _, movie_id, rating in user_data
    ]



    # Step 5: Create a dictionary to store aggregated weighted ratings for each movie
    weighted_ratings = {}
    for user_id, movie_id, rating in all_users_data:
        weighted_ratings[movie_id] = weighted_ratings.get(movie_id, 0) + rating

    # Step 6: Identify the top-rated movies based on the weighted aggregation method
    top_movies = sorted(weighted_ratings, key=weighted_ratings.get, reverse=True)[:10]

    result = [(movie_id, min(weighted_ratings[movie_id], 5.0)) for movie_id in top_movies]


    # Step 7: Return the top-rated movies and their weighted ratings
    #return [(movie_id, weighted_ratings[movie_id]) for movie_id in top_movies]
    return result





user1 = 3
user2 = 5
user3 = 10


user1_data = movie_predictor(user1)
user2_data = movie_predictor(user2)
user3_data = movie_predictor(user3)



all_users_data = user1_data + user2_data + user3_data



average_aggeregation = group_recommendations_average(all_users_data)
least = group_recommendations_least_misery(all_users_data)

print("Group Recommendations - Average Aggeregation:")
for movie_id, average_rating in average_aggeregation:
    print(f"Movie {movie_id}: Average Rating - {round(average_rating,2)}")

"""
Group Recommendations - Average Aggeregation:
Movie 306: Average Rating - 4.96
Movie 1293: Average Rating - 4.89
Movie 1612: Average Rating - 4.89
Movie 359: Average Rating - 4.85
Movie 360: Average Rating - 4.85
Movie 1025: Average Rating - 4.46
Movie 984: Average Rating - 4.46
Movie 752: Average Rating - 4.42
Movie 754: Average Rating - 4.4
Movie 313: Average Rating - 4.38

"""

print("Group Recommendations - Least Misery:")
for movie_id, min_rating in least:
    print(f"Movie {movie_id}: Minimum Rating - {round(min_rating,2)}")


"""
Group Recommendations - Least Misery:
Movie 306: Minimum Rating - 4.92     
Movie 359: Minimum Rating - 4.85
Movie 360: Minimum Rating - 4.85
Movie 1293: Minimum Rating - 4.85
Movie 1612: Minimum Rating - 4.85
Movie 752: Minimum Rating - 4.42
Movie 754: Minimum Rating - 4.4
Movie 313: Minimum Rating - 4.38
Movie 316: Minimum Rating - 4.14
Movie 902: Minimum Rating - 4.13


"""

# PART B Output
weighted = weighted_aggregation(all_users_data)

print("Group Recommendations - Weighted Aggregation:")
for movie_id, weighted_rating in weighted:
    print(f"Movie {movie_id}: Weighted Rating - {round(weighted_rating,2)}")

"""    

Group Recommendations - Weighted Aggregation:
Movie 1293: Weighted Rating - 5.0
Movie 1612: Weighted Rating - 5.0
Movie 306: Weighted Rating - 5.0
Movie 360: Weighted Rating - 5.0
Movie 359: Weighted Rating - 4.85
Movie 1106: Weighted Rating - 4.58
Movie 752: Weighted Rating - 4.42
Movie 754: Weighted Rating - 4.4
Movie 1294: Weighted Rating - 4.04
Movie 900: Weighted Rating - 3.99

"""

