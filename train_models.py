import joblib
import matplotlib.pyplot as plt
from matplotlib import colors
from dataset_reader import *
from process_arrays import *

file_pattern = 'wildfire_data/next_day_wildfire_spread_train*'

# Data statistics for various variables
DATA_STATS = {
    'elevation': (0.0, 3141.0, 657.3003, 649.0147),  # (min_clip, max_clip, mean, standard deviation)
    'pdsi': (-6.12974870967865, 7.876040384292651, -0.0052714925, 2.6823447),
    'NDVI': (-9821.0, 9996.0, 5157.625, 2466.6677),
    'pr': (0.0, 44.53038024902344, 1.7398051, 4.482833),
    'sph': (0., 1., 0.0071658953, 0.0042835088),
    'th': (0., 360.0, 190.32976, 72.59854),
    'tmmn': (253.15, 298.94891357421875, 281.08768, 8.982386),
    'tmmx': (253.15, 315.09228515625, 295.17383, 9.815496),
    'vs': (0.0, 10.024310074806237, 3.8500874, 1.4109988),
    'erc': (0.0, 106.24891662597656, 37.326267, 20.846027),
    'population': (0., 2534.06298828125, 25.531384, 154.72331),
    'PrevFireMask': (-1., 1., 0., 1.),
    'FireMask': (-1., 1., 0., 1.)
}

side_length = 32
num_obs = 100

dataset = get_dataset(
    file_pattern,
    data_size=64,
    sample_size=side_length,
    batch_size=num_obs,
    num_in_channels=12,
    compression_type=None,
    clip_and_normalize=False,
    clip_and_rescale=False,
    random_crop=True,
    center_crop=False)

inputs, labels = next(iter(dataset))

# Extract a certain previous fire mask
prev_fire_masks = inputs[:, :, :, 11]
found_it_flag = 0
img_num = 0
while found_it_flag == 0:
    fire_mask = np.array(prev_fire_masks[img_num, :, :])
    if (np.all((fire_mask == 0))):
        img_num = img_num + 1
    elif (np.all(np.invert(fire_mask == -1))):
        test_img = fire_mask
        found_it_flag = 1
    else:
        img_num = img_num + 1

np.set_printoptions(threshold=np.inf)
print('fire mask:\n', fire_mask, '\n\n')
print('computed avg neighbor fire mask:\n', avg_neighbors(fire_mask))

# Eliminate observations with uncertain squares in the previous fire mask
prev_masks_array = np.array(inputs[:, :, :, 11])

first_find_flag = 1
count = 0
indices = []

for img_num in range(num_obs):
    fire_mask = np.array(prev_fire_masks[img_num, :, :])
    if (np.all(np.invert(fire_mask == -1))):
        count += 1
        indices.append(img_num)
        if first_find_flag == 1:
            certain_prev_fire_masks = fire_mask
            first_find_flag = 0
        else:
            certain_prev_fire_masks = np.dstack((certain_prev_fire_masks, fire_mask))

full_input_array = np.array(inputs)

for i, index in enumerate(indices):
    if i == 0:
        certain_input_array = full_input_array[index, :, :, :]
        print(certain_input_array.shape)
    elif i == 1:
        certain_input_array = np.concatenate(
            (certain_input_array[..., np.newaxis], full_input_array[index, :, :, :, np.newaxis]), axis=3)
    else:
        certain_input_array = np.concatenate((certain_input_array, full_input_array[index, :, :, :, np.newaxis]),
                                             axis=3)

full_labels = np.array(labels)

for i, index in enumerate(indices):
    if i == 0:
        certain_labels = full_labels[index, :, :, :]
        surrounding_fire_scores = avg_neighbors(full_labels[index, :, :, :])
        surrounding_fire_scores = surrounding_fire_scores[..., np.newaxis]
    else:
        certain_labels = np.concatenate((certain_labels, full_labels[index, :, :, :]), axis=2)
        avg_mat = avg_neighbors(full_labels[index, :, :, :])
        surrounding_fire_scores = np.concatenate((surrounding_fire_scores, avg_mat[..., np.newaxis]), axis=2)

print(certain_labels.shape)
print(surrounding_fire_scores.shape)

# Functions
def elim_uncertain(prev_fire_mask_batch):
    prev_masks_array = np.array(prev_fire_mask_batch)
    num_imgs, rows, cols = prev_masks_array.shape

    first_find_flag = 1
    count = 0
    indices = []

    for img_num in range(num_imgs):
        fire_mask = prev_fire_mask_batch[img_num, :, :]
        if (np.all(np.invert(fire_mask == -1))):
            count += 1
            indices.append(img_num)
            if first_find_flag == 1:
                certain_prev_fire_masks_batch = fire_mask
                first_find_flag = 0
            else:
                certain_prev_fire_masks_batch = np.dstack((certain_prev_fire_masks_batch, fire_mask))

    return certain_prev_fire_masks_batch, indices

def extract_certain_labels(certain_indices, og_labels):
    for i, index in enumerate(certain_indices):
        if i == 0:
            extracted_labels = og_labels[index, :, :, :]
        else:
            extracted_labels = np.concatenate((extracted_labels, og_labels[index, :, :, :]), axis=2)

    return extracted_labels

def avg_neighbor_batch(batch_in):
    rows, cols, batch_size = batch_in.shape
    batch_out = np.zeros((rows, cols, batch_size))
    for i in range(batch_size):
        working_arr = batch_in[:, :, i]
        avgd_arr = avg_neighbors(working_arr)
        batch_out[:, :, i] = avgd_arr

    return batch_out

# Test functions above
full_prev_fire_masks = np.array(inputs[:, :, :, 11])
full_curr_fire_masks = np.array(labels)

certain_prev_masks, certain_indices = elim_uncertain(full_prev_fire_masks)
certain_labels = extract_certain_labels(certain_indices, full_curr_fire_masks)
avg_neighbors_feat = avg_neighbor_batch(certain_prev_masks)

# Linear Regression
flat_prev_masks = []

for obs in range(certain_prev_masks.shape[2]):
    for row in range(certain_prev_masks.shape[0]):
        for col in range(certain_prev_masks.shape[1]):
            flat_prev_masks.append(certain_prev_masks[row, col, obs])
flat_prev_masks = np.array(flat_prev_masks)

flat_avg_nbrs = []
for obs in range(avg_neighbors_feat.shape[2]):
    for row in range(avg_neighbors_feat.shape[0]):
        for col in range(avg_neighbors_feat.shape[1]):
            flat_avg_nbrs.append(avg_neighbors_feat[row, col, obs])
flat_avg_nbrs = np.array(flat_avg_nbrs)

X_train = np.vstack((np.transpose(flat_prev_masks), np.transpose(flat_avg_nbrs)))
X_train = np.transpose(X_train)

flat_labels = []
for obs in range(certain_labels.shape[2]):
    for row in range(certain_labels.shape[0]):
        for col in range(certain_labels.shape[1]):
            flat_labels.append(certain_labels[row, col, obs])
flat_labels = np.array(flat_labels)

Y_train = flat_labels

variables = ["currently on fire?", "# of neighbors currently on fire"]

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

lr_fire = LinearRegression().fit(X_train, Y_train)

print('training R2-score:', np.round(r2_score(Y_train, lr_fire.predict(X_train)), 2))

scores_fire = cross_val_score(lr_fire, X_train, Y_train, cv=10, scoring='r2')

print("validation R2-scores:", np.round(scores_fire, 2))
print("average:", np.round(np.mean(scores_fire), 2))

print('LR coefficients:')
for i, coeff in enumerate(lr_fire.coef_):
    print('{0:5s}  {1:>-10.2f}'.format(variables[i], np.round(coeff, 2)))

TITLES = [
  'Elevation',
  'Wind\ndirection',
  'Wind\nvelocity',
  'Min\ntemp',
  'Max\ntemp',
  'Humidity',
  'Precip',
  'Drought',
  'Vegetation',
  'Population\ndensity',
  'Energy\nrelease\ncomponent',
  'Previous\nfire\nmask',
  'Fire\nmask'
]

n_rows = 5
n_features = inputs.shape[3]
CMAP = colors.ListedColormap(['black', 'silver', 'orangered'])
BOUNDS = [-1, -0.1, 0.001, 1]
NORM = colors.BoundaryNorm(BOUNDS, CMAP.N)

fig = plt.figure(figsize=(15,6.5))

for i in range(n_rows):
  for j in range(n_features + 1):
    plt.subplot(n_rows, n_features + 1, i * (n_features + 1) + j + 1)
    if i == 0:
      plt.title(TITLES[j], fontsize=13)
    if j < n_features - 1:
      plt.imshow(inputs[i, :, :, j], cmap='viridis')
    if j == n_features - 1:
      plt.imshow(inputs[i, :, :, -1], cmap=CMAP, norm=NORM)
    if j == n_features:
      plt.imshow(labels[i, :, :, 0], cmap=CMAP, norm=NORM)
    plt.axis('off')
plt.tight_layout()

# Save the model
joblib.dump(lr_fire, 'models/wildfire_model.pkl')

# Show a visual
plt.show()
