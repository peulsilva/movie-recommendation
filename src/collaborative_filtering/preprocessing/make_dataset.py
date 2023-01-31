import pandas as pd
import numpy as np 
from tqdm import tqdm
from typing import List

def read_ratings_df()-> pd.DataFrame:
    """Returns user_ratedmovies.dat dataframe

    Returns:
        pd.DataFrame: user_ratedmovies.dat dataframe
    """   
    return  pd.read_csv(
        "data/user_ratedmovies.dat", 
        sep='\t'
    )

def train_test_split(
    ratings_matrix : pd.DataFrame,
    test_size : float = 0.3
) -> List[pd.DataFrame]:
    """Performs train/test split

    * For each user, leave 30% of its ratings to testing
    * The other 70% of its ratings will be used to training

    Args:
        ratings_matrix (pd.DataFrame): Matrix containing ```userID```
        as index, ```movieID``` as columns and ```ratings``` as values
        test_size (float, optional): Test size. Defaults to 0.3.

    Returns:
        List[pd.DataFrame]: Train and test dataframe
    """
    train_matrix= ratings_matrix.copy()
    test_matrix= pd.DataFrame(
        index = ratings_matrix.index,
        columns= ratings_matrix.columns,
        data = 0
    )

    for user_id, ratings in tqdm(ratings_matrix.iterrows()):   
        non_null_ratings = ratings[~ratings.isna()]
        
        test_idx = np.random.choice(
            non_null_ratings.index,
            size = int(test_size * non_null_ratings.shape[0]),
            replace= False
        )

        train_matrix.loc[user_id, test_idx] = 0
        test_matrix.loc[user_id, test_idx] = ratings.loc[test_idx]

    return train_matrix, test_matrix

def read_train_test_matrix() -> List[pd.DataFrame]:
    return pd.read_pickle("data/collaborative-filtering/train_matrix.pkl"),\
        pd.read_pickle("data/collaborative-filtering/test_matrix.pkl")
