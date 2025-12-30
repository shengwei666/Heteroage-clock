# age_transform.py

import numpy as np

class AgeTransformer:
    """
    Class for transforming biological age between raw age and transformed age for model input.
    
    This transformer is used to adjust the age distribution to ensure that both the younger and older age groups 
    are appropriately represented for machine learning models.
    
    Transforms raw age to a log-transformed age for model training, and allows inverse transformation 
    back to the original age scale for interpretation.
    """
    
    def __init__(self, adult_age=20):
        """
        Initialize the AgeTransformer with an offset for adult age.

        Parameters:
        - adult_age: The threshold age to separate childhood and adulthood. Default is 20.
        """
        self.adult_age = adult_age
        self.age_offset = adult_age + 1
        self.log_offset = np.log(self.age_offset)

    def transform(self, age):
        """
        Transform raw biological age to a model-friendly representation (log transformation for younger ages,
        and linear transformation for older ages).
        
        Parameters:
        - age: A numpy array or list of biological ages to be transformed.
        
        Returns:
        - Transformed age in model-friendly format (log-transformed or scaled).
        """
        age = np.asarray(age, dtype=float)
        mask = (age <= self.adult_age)
        y = np.empty_like(age)
        
        # Log transformation for ages <= adult_age
        y[mask] = np.log(np.maximum(age[mask], 0) + 1.0) - self.log_offset
        
        # Linear transformation for ages > adult_age
        y[~mask] = (age[~mask] - self.adult_age) / self.age_offset
        
        return y

    def inverse_transform(self, y):
        """
        Inverse transformation to convert the model-friendly representation back to raw age scale.
        
        Parameters:
        - y: A numpy array or list of transformed ages to be converted back to the raw scale.
        
        Returns:
        - Raw biological age in original scale.
        """
        y = np.asarray(y, dtype=float)
        mask = (y <= 0)
        age = np.empty_like(y)
        
        # Inverse log transformation for ages <= adult_age
        age[mask] = np.exp(y[mask] + self.log_offset) - 1.0
        
        # Inverse linear transformation for ages > adult_age
        age[~mask] = y[~mask] * self.age_offset + self.adult_age
        
        return age
