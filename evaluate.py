"""
Short description - This module contains code to evaluate a model

:author: Zixi Luo
"""

from train import *
import matplotlib.pyplot as plt

def evaluate(model, x_val, y_val):
    y_hat = model.predict(x_val)
    
    rmse = np.sqrt(mean_squared_error(y_val, y_hat))
    mae = mean_absolute_error(y_val, y_hat)
    
    print("RMSE: %.2f"
      %rmse)
    
    print("MAE: %.2f"
     %mae)
    
    return y_hat, rmse, mae

def plot(y_hat, y_val, output_path):
    plt.scatter(y_hat, y_val)
    plt.title('predict price VS true price')
    plt.xlabel('predict price')
    plt.ylabel('true price')
    plt.savefig(f"{output_path}/evaluation.png")
    
    
# def main():
    
#     pickle.loads()
#     evaluation(model, )


# if __name__ == "__main__":
    
