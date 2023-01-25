import pandas as pd
from tqdm import tqdm
from typing import List

def get_movie_id(
    movie_id : pd.Series,
    movie_ratings_df : pd.DataFrame,
    first_movie_id : str
)-> List[pd.DataFrame]:
    """Given a series containing data as 

    ```
    {
        882: "2:",
        1205: "3:", 
        1487: "4:",
        ...
    }
    ```
    and the dataframe containing as columns:
    ['user_id', 'rating', 'date'], but some NaN values, such as

    ```
    { 
        ...
        882 : ["2:", NaN, NaN],
        ...
        1205 : ["3:", NaN, NaN].
        ...
        1487 : ["4:", NaN, NaN]
    },
    ```

    returns a list of dataframes containing 

    ['user_id', 'rating', 'date', 'movie_id']

    and removes these null values 

    Args:
        movie_id (pd.Series): Given a series containing data as 

        ```
        {
            882: "2:",
            1205: "3:", 
            1487: "4:",
            ...
        }
        ```

        movie_ratings_df (pd.DataFrame): dataframe containing as columns:

        ['user_id', 'rating', 'date'], but some NaN values, such as

        ```
        { 
            ...
            882 : ["2:", NaN, NaN],
            ...
            1205 : ["3:", NaN, NaN].
            ...
            1487 : ["4:", NaN, NaN]
        }
        ```

    Returns:
        List[pd.DataFrame]: 
    """
    last_idx = movie_id.index[0]
    complete_movie_ratings_list = []
    for idx, value in tqdm(movie_id.iteritems()):
        if idx == movie_id.index[0] : 
            temp_df = movie_ratings_df.loc[0: idx]
            temp_df['movie_id'] = first_movie_id
        
        else:
            temp_df = movie_ratings_df.loc[last_idx: idx]
        
            temp_df['movie_id'] = temp_df.iloc[0]\
                ['user_id']\
                .replace(':', '')

        temp_df = temp_df.dropna()

        complete_movie_ratings_list.append(temp_df)
        last_idx = idx
    
    return complete_movie_ratings_list

def parse_data (
    dirname : str
):
    """Data from 'combined_data_1.txt' is in the 
    following format

    ```
    {   
        "1:" (movie_id)
        (user_id, rating, date)
        123, 4, "12-10-2005"
        456, 3, "17-08-2006"
        ...
        "2:" (movie_id)
        (user_id, rating, date)
        426, 1, "01-05-2014"
        887, 3, "24-02-2016"
        ...
    }
    ```

    This function parses data and returns it as
    a dataframe

    ```
    {
        (user_id, rating, date, movie_id)
        123, 4, "12-10-2005", 1
        456, 3, "17-08-2006", 1
        ...
        426, 1, "01-05-2014", 2
        887, 3, "24-02-2016", 2
        ...
    }
    ``` 
    


    Args:
        dir_name (str): _description_

    Returns:
        _type_: _description_
    """
    all_data = pd.read_csv(
        dirname,
    )
    first_movie_id = all_data.columns[0]\
        .replace(':','')

    movie_ratings_df = all_data\
        .reset_index()\
        .set_axis(['user_id', 'rating', 'date'], axis= 1)

    movie_id = movie_ratings_df\
        [movie_ratings_df['rating'].isna()]\
        .user_id

    complete_movie_ratings_list = get_movie_id(
        movie_id,
        movie_ratings_df,
        first_movie_id
    )

    return pd.concat(complete_movie_ratings_list)

