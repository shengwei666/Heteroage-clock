"""
heteroage_clock.core.age_transform

Implements the Log-Linear transformation for biological age.
Based on the Horvath clock methodology:
- Logarithmic scale for young ages (development).
- Linear scale for adult ages (aging).
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class AgeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, adult_age: float = 20.0):
        """
        Args:
            adult_age (float): The age threshold where development ends and aging begins.
                               Standard value is 20.
        """
        self.adult_age = adult_age

    def fit(self, X, y=None):
        return self

    def transform(self, age: np.ndarray) -> np.ndarray:
        """
        Transform chronological age to the log-linear scale.
        
        Formula:
        - If age <= adult_age: log((age + 1) / (adult_age + 1))
        - If age > adult_age:  (age - adult_age) / (adult_age + 1)
        """
        age = np.array(age).astype(float)
        
        # Calculate transformation
        # Case 1: Young (Logarithmic)
        # Note: We usually align so that at adult_age, value is 0. 
        # But Horvath's F function typically aligns differently. 
        # Let's use the standard Horvath-style transformation logic:
        # F(age) = log(age+1) - log(adult+1)    if age <= adult
        # F(age) = (age - adult)/(adult+1)      if age > adult
        
        # Vectorized implementation
        transformed = np.zeros_like(age)
        
        mask_young = age <= self.adult_age
        mask_old = ~mask_young
        
        if np.any(mask_young):
            transformed[mask_young] = np.log(age[mask_young] + 1) - np.log(self.adult_age + 1)
            
        if np.any(mask_old):
            transformed[mask_old] = (age[mask_old] - self.adult_age) / (self.adult_age + 1)
            
        return transformed

    def inverse_transform(self, transformed_age: np.ndarray) -> np.ndarray:
        """
        Convert transformed values back to chronological age (years).
        
        Inverse Formula:
        - If y < 0: (adult_age + 1) * exp(y) - 1
        - If y >= 0: y * (adult_age + 1) + adult_age
        """
        transformed_age = np.array(transformed_age).astype(float)
        original_age = np.zeros_like(transformed_age)
        
        # Threshold 0 corresponds to adult_age in the transform above
        mask_young = transformed_age < 0
        mask_old = ~mask_young
        
        if np.any(mask_young):
            original_age[mask_young] = (self.adult_age + 1) * np.exp(transformed_age[mask_young]) - 1
            
        if np.any(mask_old):
            original_age[mask_old] = transformed_age[mask_old] * (self.adult_age + 1) + self.adult_age
            
        return original_age