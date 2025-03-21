'''
gaussian module: home for LinearRegression
'''

from scipy import stats # --for p values comp from t test strictly (all other stats are done from formulae)
from base import GLM
import numpy as np
import pandas as pd

def append_ones(X:np.ndarray):
    return np.c_[np.ones(X.shape[0]),X]

def make_column_vector(y:np.array):
    return y.reshape(-1,1)

class LinearModel(GLM):
    family='Gaussian'

    def __init__(self, X, y):
        self.__X=append_ones(X)
        self.__y=make_column_vector(y)
        self.__n, self.__p=self.__X.shape  
        self.__p-=1
        self.__beta=None
        self.__fit=False

    @property
    def X(self):
        return self.__X
    @property
    def y(self):
        return self.__y
    @X.setter
    def X(self, X):
        raise ValueError('X is immutable')
    @y.setter
    def y(self, y):
        raise ValueError('y is immutable')

    @property
    def coefficients(self):
        return self.__beta
    @property
    def y_hat(self):
        return self.predict(self.__X)

    @property
    def RSS(self):
        # --using residuals
        return np.sum((self.__residuals)**2)
    @property
    def TSS(self):
        return np.sum((self.__y-np.mean(self.__y))**2)
    @property
    def MSE(self):
        return self.RSS/self.__n
    @property
    def R2(self):
        return 1-self.RSS/self.TSS
    @property
    def adjR2(self):
        return 1-(1-self.R2)*(self.__n-1)/(self.__n-self.__p-1)


    
    # @staticmethod
    # def pdf(x):
    #     # computing y from a single x
    #     y = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((x-mu)/sigma)**2)
    #     return y
    
    def fit(self):

        self.__beta=np.linalg.inv(self.__X.T@self.__X)@self.__X.T@self.__y
        self.__y_hat=self.__X@self.__beta
        self.__residuals=self.__y-self.__y_hat
        self.__fit=True

    def predict(self, X):
        if not self.__fit:
            raise ValueError('Model not fit yet')
        return X@self.__beta
    
    def summary(self):
        if not self.__fit:
            raise ValueError('Model not fit yet')

        print('-'*50)
        print(self.__repr__())
        
        t_values, p_values, f_stat, f_p_value=self.stat()
        df=pd.DataFrame({'Coefficients':self.__beta.flatten(),
                         't_values':t_values.flatten(),
                         'p_values':p_values.flatten()},
                        index=['Intercept']+['X'+str(i) for i in range(1,self.__p+1)])
        df['Significance']=(df['p_values']<0.05).map({True:'*', False:''})
        
        print(df)

        print('-'*50)
        print(f'Residual standard error: {np.sqrt(self.MSE):.3f} on {self.__n-self.__p-1} degrees of freedom')
        print(f'Multiple R-squared: {self.R2:.5f}, Adjusted R-squared: {self.adjR2:.5f}')
        # print(f'F-statistic: {f_stat:.1f} on {self.__p} and {self.__n-self.__p-1} DF, p-value: {f_p_value}')
        print('-'*50)



    def stat(self):
        if not self.__fit:
            raise ValueError('Model not fit yet')
        sigma_2=np.sum((self.__y - self.y_hat)**2)/(self.__n-self.__p-1)
        t_values=self.__beta/ make_column_vector(np.sqrt(sigma_2 * np.linalg.diagonal(np.linalg.inv(self.__X.T @ self.__X))))
        p_values=2 * stats.t.sf(np.abs(t_values), self.__n-self.__p-1)

        f_stat=(self.RSS/self.__p)/(self.RSS/(self.__n-self.__p-1))
        f_p_value=1-stats.f.cdf(f_stat, self.__p, self.__n-self.__p-1)

        return (t_values, p_values, f_stat, f_p_value)


    def __call__(self):
        return self.fit()

    def __repr__(self): 
        return f'''LinearModel(
        family='Gaussian',
        n={self.__n},
        p={self.__p},
        )
        '''
    
    def __str__(self):
        return self.summary()
    
def main():
    # --testing on vul data
    df= pd.read_csv('example/vul.csv', index_col=0, sep=' ' )
    X=df.iloc[:,2]
    X=X.to_numpy()
    
    y=df.iloc[:,6]
    y=y.to_numpy()

    model=LinearModel(X,y)
    model.fit()
    model.summary()

    # -- successfull
    # --------------------------------------------------
    # LinearModel(
    #         family='Gaussian',
    #         n=144,
    #         p=1,
    #         )
            
    #            Coefficients  t_values  p_values Significance
    # Intercept     -1.604777 -3.802467  0.000212            *
    # X1             0.517379  3.606747  0.000429            *
    # --------------------------------------------------
    # Residual standard error: 1.670 on 142 degrees of freedom
    # Multiple R-squared: 0.08392, Adjusted R-squared: 0.07747
    # --------------------------------------------------

if __name__ == '__main__':
    main()

