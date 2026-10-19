#!/usr/bin/env python3
import math
import numpy as np

def normalize(x, x_min, x_max):
    """
    Normalize x to the range [0, 1] and clip the result to ensure it remains within [0, 1].
    
    Parameters:
      x: current value
      x_min: minimum value
      x_max: maximum value
      
    Returns:
      Ratio of x between x_min and x_max, clipped to [0,1].
    """
    if x_max == x_min:
        raise ValueError("x_max and x_min cannot be equal")
    ratio = (x - x_min) / (x_max - x_min)
    return np.clip(ratio, 0, 1)

def linear_fn(ratio):
    """
    Linear function.
    
    Parameters:
      ratio: A value between 0 and 1
      
    Returns:
      Linear mapping of ratio.
    """
    return ratio

def cosine_fn(ratio):
    """
    Cosine weighting function.
    The formula used is: y = (1 - cos(pi * ratio)) / 2
    
    Parameters:
      ratio: A value between 0 and 1
      
    Returns:
      Cosine weighted value mapped to [0,1].
    """
    return (1 - math.cos(math.pi * ratio)) / 2

def log_fn(ratio):
    """
    Logarithmic function:
    To ensure y is in [0, 1], we use the formula:
      y = log(k * ratio + 1) / log(k + 1)
    where k > 0 controls the curve shape. Here, k is set to 9 so that y=1 when ratio=1.
    
    Parameters:
      ratio: A normalized value between 0 and 1
      
    Returns:
      Logarithmically scaled value mapped to [0,1].
    """
    k = 9
    return math.log(k * ratio + 1) / math.log(k + 1)

def exp_fn(ratio):
    """
    Exponential function:
    To map y to [0, 1], we use the equation:
      y = (exp(ratio) - 1) / (e - 1)
    so that y=0 when ratio=0 and y=1 when ratio=1.
    
    Parameters:
      ratio: A normalized value between 0 and 1
      
    Returns:
      Exponentially scaled value mapped to [0,1].
    """
    return (math.exp(ratio) - 1) / (math.e - 1)

def calculate_weight(trend, fn_type, x, x_min, x_max):
    """
    Calculate the weight based on the provided trend, function type, and input values.
    
    Parameters:
      trend: string, either 'increase' (weight increases with x) or 'decrease' (weight decreases with x)
      fn_type: string, one of the following: 'linear', 'cosine', 'log', 'exp'
      x: current x value
      x_min: minimum x value
      x_max: maximum x value
      
    Returns:
      y: a weight normalized to the range [0, 1].
    """
    # Normalize x to a ratio in [0,1]
    ratio = normalize(x, x_min, x_max)

    # Compute y based on the selected function
    if fn_type == 'linear':
        y = linear_fn(ratio)
    elif fn_type == 'cosine':
        y = cosine_fn(ratio)
    elif fn_type == 'log':
        y = log_fn(ratio)
    elif fn_type == 'exp':
        y = exp_fn(ratio)
    else:
        raise ValueError("Unsupported function type. Please use 'linear', 'cosine', 'log', or 'exp'.")

    # Adjust weight based on the trend
    if trend == 'increase':
        y = 1 - y
    elif trend != 'decrease':  # If trend is neither 'increase' nor 'decrease', raise an error.
        raise ValueError("Trend must be either 'increase' or 'decrease'")

    return y

# Main block for testing the functionality
if __name__ == '__main__':
    # Sample input values
    trend = 'increase'   # Options: 'increase' or 'decrease'
    fn_type = 'cosine'   # Options: 'linear', 'cosine', 'log', 'exp'
    x_min = 0
    x_max = 10
    x = 5

    weight = calculate_weight(trend, fn_type, x, x_min, x_max)
    print("Trend: {}, Function: {}, Current x: {} , x Range: [{} - {}] => Weight y = {:.4f}"
          .format(trend, fn_type, x, x_min, x_max, weight))
