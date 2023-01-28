import pandas as pd
from typing import Dict

def get_movies_id_map() -> Dict:
    """Returns movies id->name map 
    ```
    {
        1: 'Toy story',
        2: 'Jumanji',
        3: 'Grumpy Old Men',
        4: 'Waiting to Exhale',
        5: 'Father of the Bride Part II',
        ...
    }
    ```

    Returns:
        Dict: _description_
    """    

    movies = pd.read_csv(
        "data/movies.dat", 
        sep='\t',
        encoding='latin1'
    )

    return movies.set_index("id")\
        .title\
        .to_dict()