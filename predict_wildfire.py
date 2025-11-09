import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib import colors

# Load the trained model
lr_model = joblib.load('models/wildfire_model.pkl')


# Function to input previous fire mask and get predictions
def predict_wildfire(prev_fire_mask, avg_neighbors):
    # Convert input variables into a format suitable for prediction
    input_data = np.array([[prev_fire_mask, avg_neighbors]])

    # Make prediction using the loaded model
    prediction = lr_model.predict(input_data)

    return prediction


# Example input values
prev_fire_mask = 4  # Assuming no previous fire
avg_neighbors = 1  # Assuming 1 neighbor has fire

# Get prediction
prediction = predict_wildfire(prev_fire_mask, avg_neighbors)

# Create a visual representation of the predicted fire spread
CMAP = colors.ListedColormap(['black', 'silver', 'orangered'])
BOUNDS = [-1, -0.1, 0.001, 1]
NORM = colors.BoundaryNorm(BOUNDS, CMAP.N)

fig, ax = plt.subplots(figsize=(6, 6))
plt.imshow([[prediction]], cmap=CMAP, norm=NORM)
plt.title("Predicted Fire Spread")
plt.colorbar(label="Predicted Fire Spread (Units)")
plt.axis('off')
plt.show()
