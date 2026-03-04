import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. The Mock Data (Simulating a Lean Bulk)
# We are tracking days passed and morning weigh-ins
data = {
    'Day': [1, 5, 10, 15, 20, 25, 30],
    'Weight_kg': [74.0, 74.2, 74.5, 74.6, 75.0, 75.1, 75.4]
}
df = pd.DataFrame(data)

# 2. Prepare Data for Scikit-Learn
# ML models need 'X' (the inputs) and 'y' (the target we want to predict)
# Scikit-learn requires 'X' to be a 2D structure (like a DataFrame), hence the double brackets.
X = df[['Day']] 
y = df['Weight_kg']

# 3. Create and Train the Model (The ML Magic)
model = LinearRegression()
model.fit(X, y) # "fit" means "learn from the data"

# 4. Predict the Future!
# Let's ask the model to guess your weight for the next 100 days
future_days = pd.DataFrame({'Day': range(31, 131)})
future_weights = model.predict(future_days)

# 5. Extract the Math (y = mx + b)
# model.coef_[0] is the slope (kg gained per day)
# model.intercept_ is the starting weight (Day 0)
slope = model.coef_[0]
intercept = model.intercept_

# Calculate exactly when you hit 80kg: (Target - Start) / Rate of Gain
days_to_80 = (80 - intercept) / slope
print(f"Based on current trends, you are gaining {slope:.3f}kg per day.")
print(f"-> You will hit 80kg on Day {int(days_to_80)}!")

# 6. Visualize the Forecast
plt.figure(figsize=(10, 6))

# Plot the real data we trained on
plt.scatter(df['Day'], df['Weight_kg'], color='blue', s=100, label='Actual Weigh-ins')

# Plot the AI's prediction line
plt.plot(future_days['Day'], future_weights, color='red', linestyle='--', label='AI Forecast')

# Draw a line for the 80kg goal
plt.axhline(y=80, color='green', linestyle=':', label='Target Goal (80kg)')

plt.title('Lean Bulk Progression & Forecast')
plt.xlabel('Days into Bulk')
plt.ylabel('Body Weight (kg)')
plt.legend()
plt.grid(True)

# Save the image so we can view it in WSL
plt.savefig('gym_forecast.png')
print("Graph saved as gym_forecast.png")