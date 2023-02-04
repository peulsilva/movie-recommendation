import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List
from sklearn.base import BaseEstimator

class SGDOptimizer(BaseEstimator):
    """
        Model that is fitted by Stochastic Gradient Descent (SGD) on
        cost function (squared error).
        
        PS.: Regularization can be 'l1' or 'l2'
    """    
    def __init__(
        self, 
        regularization : str = 'l2',
        regularization_factor : float = 0.1,
        learning_rate : float = 0.001,
        num_features : float = 5,
        train_epochs : int = 100,
        epsilon : float = 1e-5,
        predict_constraints : bool = False
    ) -> None:
        
        self.regularization = regularization
        self.num_features = num_features
        self.regularization_factor = regularization_factor
        self.train_epochs = train_epochs
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.predict_constraints = predict_constraints
        self.train_errors = []
        self.val_errors = []

    def rms(
        self, 
        y_true : np.ndarray, 
        y_pred : np.ndarray
    ) -> float:
        """Calculate the root mean squared error of the non
        null predicions

        Args:
            y_true (np.ndarray): _description_
            y_pred (np.ndarray): _description_

        Returns:
            float: root mean squared error
        """        
        non_zero = y_true.nonzero()
        err = (y_true[non_zero] - y_pred[non_zero])**2

        return np.sqrt(np.mean(err))
    
    def fit(
        self, 
        y_train : np.ndarray,
        y_val : np.ndarray
    ):
        """Trains and returns X and Theta matrices with SGD

        Args:
            y_train (np.ndarray): matrix containing 
            index = user_id
            columns = movie_id
            values = rating[user_id, movie_id]
            
        """        
        users, movies = np.nonzero(y_train)
        n_users, n_movies = y_train.shape

        self.Theta = 5 * np.random.rand(self.num_features, n_users)
        self.X =5 * np.random.rand(self.num_features, n_movies)

        if self.regularization == 'l2':
            gradient = self.__l2_gradient
        
        elif self.regularization == 'l1':
            gradient = self.__l1_gradient

        for train_epoch in range(self.train_epochs):
            if train_epoch > 5:
                if abs(self.train_errors[-1] - self.train_errors[-2]) < self.epsilon:
                    break

            for u, m in tqdm(zip(users, movies)):
                y_pred = self.Theta[:, u].T @ self.X[:, m]

                if self.predict_constraints == True:
                    if y_pred > 5 :
                        y_pred = 5
                    elif y_pred < 0 :
                        y_pred = 0

                error = (y_train[u,m] - y_pred)

                dTheta_u, dX_m = gradient(u, m, error)

                self.Theta[:, u] += self.learning_rate * dTheta_u
                self.X[:, m] += self.learning_rate * dX_m 
                
            y_pred = self.predict()
            train_error = self.rms(y_train, y_pred)
            val_error = self.rms(y_val, y_pred)

            self.train_errors.append(train_error)
            self.val_errors.append(val_error)

            print(f'train epoch {train_epoch}; \ntrain error: {train_error} \nvalidation error: {val_error}')

    def __l2_gradient(
        self,
        user_idx : int,
        movie_idx : int,
        error : float
    ):
        dTheta_u = error * self.X[:, movie_idx]\
             - self.regularization_factor*(self.Theta[:, user_idx]) 

        dX_m = error * self.Theta[:, user_idx]\
             - self.regularization_factor*(self.X[:, movie_idx]) 

        return dTheta_u, dX_m

    def __l1_gradient(
        self,
        user_idx : int,
        movie_idx : int,
        error : float
    ):
        dTheta_u = error * self.X[:, movie_idx]\
             - self.regularization_factor* np.sign(self.Theta[:, user_idx]) 

        dX_m = error * self.Theta[:, user_idx]\
             - self.regularization_factor* np.sign(self.X[:, movie_idx])

        return dTheta_u, dX_m 
    
    def predict(
        self,
        Theta : np.ndarray = None,
        X : np.ndarray = None
    ) -> np.ndarray:
        """Return predictions matrix as theta.T @ X

        Returns:
            np.ndarray: _description_
        """   
        if (Theta is not None) and (X is not None):
            self.Theta = Theta
            self.X = X
        y_pred = self.Theta.T @ self.X

        if self.predict_constraints :
            y_pred[y_pred > 5] = 5
            y_pred[y_pred < 0] = 0
        
        return y_pred

    def get_evaluation_errors(self) -> List[float]:
        return self.train_errors, self.val_errors
        # return Theta, X

    def reshape(self, matrix: np.ndarray):
        if len(matrix.shape) > 1 :
            return matrix
        
        new_shape = (
            self.num_features, 
            int(matrix.shape[0]/self.num_features)
        )

        return np.reshape(matrix, new_shape)
            