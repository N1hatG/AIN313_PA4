import os
import numpy as np
from sklearn.model_selection import train_test_split
from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesScalerMinMax
from sklearn.model_selection import train_test_split
from tslearn.svm import TimeSeriesSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time

DATA_PATH = "C:/Users/Umay/Downloads/poses_npz/poses_npz/" 
TEST_SIZE = 0.2
RANDOM_SEED = 42

def load_data_recursive(root_folder):
    X_list = []
    y_list = []
    max_length = 0
    file_count = 0
    
    print(f"{root_folder}")
    
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.npz'):
                file_path = os.path.join(root, file)
                try:
                    data = np.load(file_path, allow_pickle=True)
                    
                    features = data['pose_norm']  # Normalized data
                    label = data['label']         # label (0 to 5 numbers)
                    
                    if features.shape[0] == 0: 
                        continue
                        
                    # Flattening
                    # input: (frames_count, 25, 3) -> output: (frames_count, 75)
                    frames_count = features.shape[0]
                    flattened_features = features.reshape(frames_count, -1)
                    
                    # max length video (for padding)
                    if frames_count > max_length:
                        max_length = frames_count
                    
                    X_list.append(flattened_features)
                    y_list.append(label)
                    file_count += 1
                    
                except Exception as e:
                    print(f"error ({file}): {e}")

    return X_list, np.array(y_list), max_length

X_raw, y, max_len = load_data_recursive(DATA_PATH)

if len(X_raw) == 0:
    print("file couldnt be found")
else:
    print(f"max length video: {max_len} frame")

    # MANUAL PADDING (NUMPY) 
    print("Padding process")
    
    num_samples = len(X_raw)
    num_features = X_raw[0].shape[1] 
    

    X_padded = np.zeros((num_samples, max_len, num_features), dtype='float32')
    
    for i, sequence in enumerate(X_raw):
        length = len(sequence)
        X_padded[i, :length, :] = sequence

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_padded, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )


# Downsampling 
# GAK, has N^2 complexity.
# with List comprehension only taking every 5th frame of each video
DOWNSAMPLE_RATE = 4  
X_list_small = [x[::DOWNSAMPLE_RATE] for x in X_raw]


# convert tslearn format (instead of padding)
X_formatted = to_time_series_dataset(X_list_small)

# Scaling - important for SVM 
# 0-1  normalization
scaler = TimeSeriesScalerMinMax()
X_scaled = scaler.fit_transform(X_formatted)

# Train/Test Split 
X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# GAK + SVM training (with wider gamma range)
# leaving gamma="auto" is best; tslearn scales according to the data.
print("\nGAK+SVM training is starting (without padding)")
start_time = time.time()

# increasing the C parameter a bit to strengthen the penalty, leaving gamma on automatic.
clf = TimeSeriesSVC(kernel="gak", C=10.0, gamma="auto", verbose=1)
clf.fit(X_train_g, y_train_g)

# Test
y_pred_g = clf.predict(X_test_g)
acc_g = accuracy_score(y_test_g, y_pred_g)

print(f"\nTEST ACCURACY: %{acc_g * 100:.2f}")

# visualization
cm = confusion_matrix(y_test_g, y_pred_g)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Box', 'Clap', 'Wave', 'Jog', 'Run', 'Walk'],
            yticklabels=['Box', 'Clap', 'Wave', 'Jog', 'Run', 'Walk'])
plt.title(f'GAK+SVM Accuracy: %{acc_g*100:.2f}')
plt.xlabel('prediction')
plt.ylabel('real value')
plt.show()
