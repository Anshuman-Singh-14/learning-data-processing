import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



# 1. Create Synthetic F1 Dataset
# Variables using camelCase
np.random.seed(42)
nRaces = 200
gridPos = np.random.randint(1, 21, nRaces)
pitStopTime = np.random.uniform(2.0, 5.0, nRaces)
tyreAge = np.random.randint(1, 30, nRaces)



# Target: Final Position (with some random noise)
# Formula: Final ~ 0.8*Grid + 0.5*Pit + 0.1*Tyre + Noise
finalPos = (0.8 * gridPos + 0.5 * pitStopTime + 0.1 * tyreAge + np.random.normal(0, 1, nRaces))
finalPos = np.clip(np.round(finalPos), 1, 20) 


df = pd.DataFrame({
    'gridPos': gridPos,
    'pitStopTime': pitStopTime,
    'tyreAge': tyreAge,
    'finalPos': finalPos
})


# 2. Split Features and Target
xData = df[['gridPos', 'pitStopTime', 'tyreAge']]
yData = df['finalPos']


xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.2, random_state=42)



# 3. Train Multiple Linear Regression Model
mlrModel = LinearRegression()
mlrModel.fit(xTrain, yTrain)


# 4. Predictions and Evaluation
yPred = mlrModel.predict(xTest)

print(f"Mean Squared Error: {mean_squared_error(yTest, yPred):.4f}")
print(f"R2 Score: {r2_score(yTest, yPred):.4f}")
print(f"Coefficients: {mlrModel.coef_}")



# 5. Visualizing Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(yTest, yPred, color='red', edgecolors='k', alpha=0.7, label='Actual vs Predicted')
plt.plot([1, 20], [1, 20], 'k--', lw=2, label='Identity Line')
plt.xlabel('Actual Final Position')
plt.ylabel('Predicted Final Position')
plt.title('F1 Race Results: Actual vs Predicted (Multiple Linear Regression)')
plt.legend()
plt.grid()


























plt.grid(True)
plt.savefig('f1_regression_results.png')
plt.show()