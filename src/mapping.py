import pandas as pd
from typing import Dict

def get_movie_id_name_map()-> Dict:
    """Returns a dictionary with movie_id -> movie_name

    Returns:
        Dict: _description_
    """
    movie_titles= pd.read_csv(
        "data/movie_titles.csv", 
        encoding="latin-1",
        on_bad_lines='skip',
        names=["movie_id", "year", "name"]
    )

    return movie_titles.set_index("movie_id")\
        .name\
        .to_dict()