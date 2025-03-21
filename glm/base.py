'''
base module: home for GLM interface
'''

from abc import abstractmethod, ABC


class GLM(ABC):
    '''
    Generalized Linear Model interface

    *enforces abstract behaviors on all GLM subclasses*

    inherited by: 
        * LinearModel
        * LogisticModel
        * PoissonModel
    '''

    def __init__(self, X, y):
        pass


    @abstractmethod
    def fit(self):
        raise NotImplementedError
    @abstractmethod
    def predict(self):
        raise NotImplementedError
    @abstractmethod
    def summary(self):
        raise NotImplementedError
    
    @abstractmethod
    def __call__(self):
        raise NotImplementedError
    @abstractmethod
    def __repr__(self):
        raise NotImplementedError
    @abstractmethod
    def __str__(self):
        raise NotImplementedError