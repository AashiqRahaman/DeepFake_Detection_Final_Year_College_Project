import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    BatchNormalization,
    Dropout,
    Reshape,
    Add,
    Flatten,
)
from tensorflow.keras.models import Model
import tensorflow.keras.applications.densenet as densenet
import tensorflow.keras.applications.efficientnet as efficientnet
import tensorflow.keras.applications.xception as xception
import tensorflow.keras.preprocessing.image as tf_image

import numpy as np
import os
import glob
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import warnings
import argparse

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)

# --- Custom Attention Layers (from deepfake-detection2/model.py) ---
# These layers are necessary for the attention mechanism.


class ModifiedBranch(tf.keras.layers.Layer):
    """
    Computes the modified branch for the attention mechanism.
    (From deepfake-detection2/model.py)
    """

    def __init__(self, a_vec_size, **kwargs):
        super(ModifiedBranch, self).__init__(**kwargs)
        self.a_vec_size = a_vec_size
        self.dense_layer = Dense(self.a_vec_size, activation="tanh", name="att_mod_dense")

    def call(self, input):
        af = tf.keras.backend.mean(input, axis=2)
        hs = self.dense_layer(af)
        return hs

    def get_config(self):
        config = super().get_config()
        config.update({"a_vec_size": self.a_vec_size})
        return config


class MainBranch(tf.keras.layers.Layer):
    """
    Computes the main branch for the attention mechanism.
    (From deepfake-detection2/model.py)
    """

    def __init__(self, a_vec_size, dim, **kwargs):
        super(MainBranch, self).__init__(**kwargs)
        self.a_vec_size = a_vec_size
        self.dim = dim
        self.reshape1 = Reshape((-1, self.a_vec_size), name="att_main_reshape1")
        self.relu = tf.keras.activations.relu
        self.dropout = Dropout(0.5, name="att_main_dropout")
        self.reshape2 = Reshape((self.dim**2, self.a_vec_size), name="att_main_reshape2")

    def call(self, input):
        e = tf.transpose(input, perm=[0, 2, 1])
        e = self.reshape1(e)
        e = self.relu(e)
        e = self.dropout(e)
        e = self.reshape2(e)
        e = tf.transpose(e, perm=[0, 2, 1])
        return e

    def get_config(self):
        config = super().get_config()
        config.update({"a_vec_size": self.a_vec_size, "dim": self.dim})
        return config


class Attention(tf.keras.layers.Layer):
    """
    Implements the attention technique on the two branches.
    (From deepfake-detection2/model.py)
    """

    def __init__(self, dim, a_vec_size, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.dim = dim
        self.a_vec_size = a_vec_size
        self.dense1 = Dense(self.dim**2, name="att_att_dense1")
        self.reshape1 = Reshape((1, self.dim**2), name="att_att_reshape1")
        self.add = Add(name="att_att_add")
        self.dropout = Dropout(0.5, name="att_att_dropout")
        self.relu = tf.keras.activations.relu
        self.reshape2 = Reshape((-1, self.a_vec_size), name="att_att_reshape2")
        self.dense2 = Dense(1, use_bias=False, name="att_att_dense2")
        self.reshape3 = Reshape((-1, self.dim**2), name="att_att_reshape3")

    def call(self, input):
        # input[0] is modified branch, input[1] is main branch
        eh = self.dense1(input[0])
        eh = self.reshape1(eh)
        eh = self.add([input[1], eh])
        eh = self.relu(eh)
        eh = self.dropout(eh)
        eh = tf.transpose(eh, perm=[0, 2, 1])
        eh = self.reshape2(eh)
        eh = self.dense2(eh)
        eh = self.reshape3(eh)
        eh = self.relu(eh)
        # The output 'eh' is the attention-weighted feature map (flattened)
        # Original model had a Dense(2, 'softmax') here.
        # We return the features *before* that final classification.
        return eh

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim, "a_vec_size": self.a_vec_size})
        return config


# --- Combined Detector Class (merging both notebooks) ---


class CombinedDeepfakeDetector:
    def __init__(self, input_shape=(299, 299, 3)):
        print("Initializing CombinedDeepfakeDetector...")
        self.input_shape = input_shape
        self.scaler = StandardScaler()
        self.knn_classifier = KNeighborsClassifier(n_neighbors=5)
        self.feature_extractors = {}

        # Create the three attention-based feature extractors
        for name in ["DenseNet121", "EfficientNetB0", "Xception"]:
            print(f"Creating attention extractor for: {name}")
            self.feature_extractors[name] = self.create_attention_extractor(name)

    def get_backbone_config(self, backbone_name):
        """
        Returns the correct pre-trained backbone, preprocessing function,
        feature map layer, and dimensions.
        """
        if backbone_name == "DenseNet121":
            base_model = densenet.DenseNet121(
                include_top=False, weights="imagenet", input_shape=self.input_shape
            )
            # Output map is (None, 9, 9, 1024)
            feature_map_layer = base_model.layers[-2].output
            preprocess_func = densenet.preprocess_input
            dim = 9
            a_vec_size = 1024
        elif backbone_name == "EfficientNetB0":
            base_model = efficientnet.EfficientNetB0(
                include_top=False, weights="imagenet", input_shape=self.input_shape
            )
            # Output map is (None, 9, 9, 1280)
            feature_map_layer = base_model.layers[-3].output
            preprocess_func = efficientnet.preprocess_input
            dim = 9
            a_vec_size = 1280
        elif backbone_name == "Xception":
            base_model = xception.Xception(
                include_top=False, weights="imagenet", input_shape=self.input_shape
            )
            # Output map is (None, 10, 10, 2048) in newer TF,
            # but notebook 2 used layer -13 which is (None, 19, 19, 1024)
            feature_map_layer = base_model.layers[-13].output
            preprocess_func = xception.preprocess_input
            dim = 19
            a_vec_size = 1024
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        base_model.trainable = False
        return base_model, preprocess_func, feature_map_layer, dim, a_vec_size

    def create_attention_extractor(self, backbone_name):
        """
        Creates a model that applies the attention mechanism (from notebook 2)
        to the feature maps of the specified backbone (from notebook 1).
        """
        base_model, _, feature_map_layer, dim, a_vec_size = (
            self.get_backbone_config(backbone_name)
        )

        # This logic is from the `model` function in deepfake-detection2/model.py
        # We apply it to the feature map layer of the backbone
        x = Conv2D(
            filters=a_vec_size,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            use_bias=True,
            name=f"{backbone_name}_att_conv",
        )(feature_map_layer)
        x = BatchNormalization(axis=-1, name=f"{backbone_name}_att_bn")(x)
        x = tf.keras.activations.relu(x)
        x = Dropout(0.8, name=f"{backbone_name}_att_drop")(x)
        x = Reshape((a_vec_size, dim**2), name=f"{backbone_name}_att_reshape")(x)

        # Apply the custom attention layers
        modified = ModifiedBranch(a_vec_size, name=f"{backbone_name}_mod_branch")(x)
        main = MainBranch(a_vec_size, dim, name=f"{backbone_name}_main_branch")(x)
        attention_features = Attention(dim, a_vec_size, name=f"{backbone_name}_attention")(
            [modified, main]
        )

        # Flatten the attention output to get our 1D feature vector
        output_features = Flatten(name=f"{backbone_name}_flatten")(attention_features)

        # Create the final extractor model
        extractor = Model(
            inputs=base_model.input,
            outputs=output_features,
            name=f"{backbone_name}_AttentionExtractor",
        )
        return extractor

    def extract_features(self, file_paths, batch_size=32):
        """
        Extracts features from all three attention-based extractors
        and stacks them.
        """
        all_features = {name: [] for name in self.feature_extractors.keys()}

        # Get preprocessing functions
        preprocessors = {
            "DenseNet121": densenet.preprocess_input,
            "EfficientNetB0": efficientnet.preprocess_input,
            "Xception": xception.preprocess_input,
        }

        for i in range(0, len(file_paths), batch_size):
            if i % (batch_size * 10) == 0:
                print(f"  Processing batch {i // batch_size} / {len(file_paths) // batch_size}")
                
            batch_paths = file_paths[i : i + batch_size]
            
            # Create a batch for each model type due to different preprocessing
            batch_images = {}
            
            # Load images once
            loaded_images = []
            valid_paths_idx = []
            for idx, img_path in enumerate(batch_paths):
                try:
                    image = tf_image.load_img(
                        img_path, target_size=self.input_shape[:2]
                    )
                    image = tf_image.img_to_array(image)
                    loaded_images.append(image)
                    valid_paths_idx.append(idx)
                except Exception as e:
                    print(f"Warning: Error loading {img_path}: {e}")
                    
            if not loaded_images:
                continue

            loaded_images = np.array(loaded_images)

            # Preprocess and extract for each model
            for name, extractor in self.feature_extractors.items():
                preprocess_func = preprocessors[name]
                # Preprocess the batch
                preprocessed_batch = preprocess_func(loaded_images.copy())
                
                # Get features
                features = extractor.predict(preprocessed_batch, verbose=0)
                all_features[name].extend(features)

        # Stack features horizontally
        print("Stacking extracted features...")
        stacked_features = np.concatenate(
            [
                np.array(all_features["DenseNet121"]),
                np.array(all_features["EfficientNetB0"]),
                np.array(all_features["Xception"]),
            ],
            axis=1,
        )

        return stacked_features

    # --- Feature Selection (from feature-selection-aided...ipynb) ---

    def relief_f_score(self, X, y, k=10):
        """ReliefF feature selection algorithm"""
        print("  Running ReliefF...")
        n_samples, n_features = X.shape
        feature_scores = np.zeros(n_features)

        for i in range(n_samples):
            if i % 100 == 0:
                print(f"    ReliefF processing sample {i} / {n_samples}")
            distances = np.sum((X - X[i]) ** 2, axis=1)
            nearest_indices = np.argsort(distances)[1 : k + 1]
            hits = [idx for idx in nearest_indices if y[idx] == y[i]]
            misses = [idx for idx in nearest_indices if y[idx] != y[i]]

            for j in range(n_features):
                if hits:
                    hit_diff = np.mean([abs(X[i, j] - X[idx, j]) for idx in hits])
                    feature_scores[j] -= hit_diff
                if misses:
                    miss_diff = np.mean([abs(X[i, j] - X[idx, j]) for idx in misses])
                    feature_scores[j] += miss_diff
        
        # Normalize scores
        min_score = np.min(feature_scores)
        max_score = np.max(feature_scores)
        if max_score == min_score:
            return np.zeros_like(feature_scores)
        return (feature_scores - min_score) / (max_score - min_score)

    def mrmr_score(self, X, y, selected_features=None):
        """Minimum Redundancy Maximum Relevance score helper"""
        if selected_features is None:
            selected_features = []

        mi_scores = mutual_info_classif(X, y, random_state=42)

        if not selected_features:
            return mi_scores

        n_features = X.shape[1]
        mrmr_scores = np.zeros(n_features)

        for i in range(n_features):
            if i in selected_features:
                mrmr_scores[i] = -np.inf
                continue
            
            relevance = mi_scores[i]
            
            if selected_features:
                # This is a simplified proxy for redundancy
                redundancy = np.mean(
                    [
                        mutual_info_classif(X[:, [i, j]], y, random_state=42)[0]
                        for j in selected_features
                    ]
                )
            else:
                redundancy = 0
            
            mrmr_scores[i] = relevance - redundancy
        return mrmr_scores

    def feature_selection(
        self, X_train, y_train, X_val, y_val, tau=0.3, alpha=0.1, beta=0.1, max_iterations=250
    ):
        """Combined feature selection with ReliefF, MI, mRMR and inclusion-exclusion"""
        print("Starting feature selection...")
        
        # Step 1: Calculate individual scores
        relief_scores = self.relief_f_score(X_train, y_train)
        
        print("  Running Mutual Information...")
        mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
        
        print("  Running mRMR...")
        mrmr_scores = self.mrmr_score(X_train, y_train)

        # Normalize scores
        mi_scores = (mi_scores - np.min(mi_scores)) / (np.max(mi_scores) - np.min(mi_scores))
        mrmr_scores = (mrmr_scores - np.min(mrmr_scores)) / (np.max(mrmr_scores) - np.min(mrmr_scores))

        # Step 2: Combine scores
        combined_scores = (relief_scores + mi_scores + mrmr_scores) / 3
        
        # Step 3: Initial feature selection
        n_features = len(combined_scores)
        n_initial = int(tau * n_features)
        initial_indices = np.argsort(combined_scores)[-n_initial:]

        print(f"  Initial selection: {len(initial_indices)} features")
        
        # Step 4: Inclusion-exclusion optimization
        best_features = initial_indices.copy()
        best_fitness = self.calculate_fitness(X_train, y_train, X_val, y_val, best_features) # Use X_train for fit
        print(f"  Initial fitness: {best_fitness:.4f}")

        for iteration in range(max_iterations):
            current_features = best_features.copy()
            
            # Exclusion
            n_exclude = max(1, int(alpha * len(current_features)))
            if len(current_features) > n_exclude:
                exclude_indices = np.random.choice(
                    len(current_features), n_exclude, replace=False
                )
                current_features = np.delete(current_features, exclude_indices)
            
            # Inclusion
            remaining_features = np.setdiff1d(np.arange(n_features), current_features)
            if len(remaining_features) > 0:
                n_include = min(int(beta * n_features), len(remaining_features))
                remaining_scores = combined_scores[remaining_features]
                include_indices = remaining_features[
                    np.argsort(remaining_scores)[-n_include:]
                ]
                current_features = np.unique(np.concatenate([current_features, include_indices]))
            
            current_fitness = self.calculate_fitness(X_train, y_train, X_val, y_val, current_features)

            if current_fitness > best_fitness:
                best_features = current_features.copy()
                best_fitness = current_fitness
                print(
                    f"    Iter {iteration}: New best fitness = {best_fitness:.4f}, Features = {len(best_features)}"
                )
        
        print(f"Feature selection completed. Selected {len(best_features)} features.")
        return best_features

    def calculate_fitness(self, X_train, y_train, X_val, y_val, feature_indices, weight=0.9):
        """Calculate fitness function for feature selection"""
        if len(feature_indices) == 0:
            return 0
        
        X_train_sel = X_train[:, feature_indices]
        X_val_sel = X_val[:, feature_indices]
        
        # Scale based on *training* data
        temp_scaler = StandardScaler()
        X_train_scaled = temp_scaler.fit_transform(X_train_sel)
        X_val_scaled = temp_scaler.transform(X_val_sel)
        
        temp_knn = KNeighborsClassifier(n_neighbors=5)
        temp_knn.fit(X_train_scaled, y_train)
        # Evaluate on *validation* data
        accuracy = temp_knn.score(X_val_scaled, y_val)
        
        feature_ratio = len(feature_indices) / X_train.shape[1]
        fitness = weight * accuracy + (1 - weight) * (1 - feature_ratio)
        return fitness

    def train_knn_classifier(self, X_train, y_train, selected_features):
        """Train KNN classifier on selected features"""
        print("Training final KNN classifier...")
        X_selected = X_train[:, selected_features]
        X_scaled = self.scaler.fit_transform(X_selected)
        self.knn_classifier.fit(X_scaled, y_train)

    def predict(self, X_test, selected_features):
        """Make predictions using trained KNN classifier"""
        print("Making predictions...")
        X_selected = X_test[:, selected_features]
        X_scaled = self.scaler.transform(X_selected)
        predictions = self.knn_classifier.predict(X_scaled)
        probabilities = self.knn_classifier.predict_proba(X_scaled)
        return predictions, probabilities


# --- Data Loading Helper (from feature-selection-aided...ipynb) ---


def prepare_dataset_paths(base_path, train_counts, val_counts, test_counts):
    """Prepare dataset paths and labels."""
    # This example assumes a structure like:
    # base_path/train/real/*.png
    # base_path/train/fake/*.png
    # ... and so on for val/test
    print(f"Loading data paths from: {base_path}")
    
    def get_paths_labels(split, real_count, fake_count):
        real_paths = sorted(glob.glob(os.path.join(base_path, split, "real", "*.*")))[:real_count]
        fake_paths = sorted(glob.glob(os.path.join(base_path, split, "fake", "*.*")))[:fake_count]
        
        paths = real_paths + fake_paths
        labels = [0] * len(real_paths) + [1] * len(fake_paths)
        
        paths, labels = shuffle(paths, labels, random_state=42)
        print(f"  Loaded {split}: {len(real_paths)} real, {len(fake_paths)} fake. Total: {len(paths)}")
        return paths, labels

    train_paths, train_labels = get_paths_labels("train", train_counts['real'], train_counts['fake'])
    val_paths, val_labels = get_paths_labels("val", val_counts['real'], val_counts['fake'])
    test_paths, test_labels = get_paths_labels("test", test_counts['real'], test_counts['fake'])
    
    return {
        "train": (train_paths, np.array(train_labels)),
        "val": (val_paths, np.array(val_labels)),
        "test": (test_paths, np.array(test_labels)),
    }


# --- Main execution block ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combined Attention + Feature Selection Deepfake Detector"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Base path to dataset (must contain train/val/test subfolders)",
    )
    # Example counts based on your notebook, update as needed
    parser.add_argument("--train_real", type=int, default=2930)
    parser.add_argument("--train_fake", type=int, default=2946)
    parser.add_argument("--val_real", type=int, default=200)
    parser.add_argument("--val_fake", type=int, default=200)
    parser.add_argument("--test_real", type=int, default=100)
    parser.add_argument("--test_fake", type=int, default=100)
    
    args = parser.parse_args()

    # --- Dataset Counts (from your notebooks) ---
    # Using FF++ counts as an example
    train_counts = {'real': args.train_real, 'fake': args.train_fake}
    val_counts = {'real': args.val_real, 'fake': args.val_fake}
    test_counts = {'real': args.test_real, 'fake': args.test_fake}

    # 1. Load Data Paths
    data = prepare_dataset_paths(args.data_path, train_counts, val_counts, test_counts)
    train_paths, train_labels = data["train"]
    val_paths, val_labels = data["val"]
    test_paths, test_labels = data["test"]

    # 2. Initialize Detector (this creates the 3 extractors)
    detector = CombinedDeepfakeDetector(input_shape=(299, 299, 3))

    # 3. Extract features for all splits
    print("\nExtracting Training Features...")
    train_features = detector.extract_features(train_paths)
    print("\nExtracting Validation Features...")
    val_features = detector.extract_features(val_paths)
    print("\nExtracting Test Features...")
    test_features = detector.extract_features(test_paths)
    
    print(f"\nStacked feature dimensions: {train_features.shape[1]}") # Should be 523

    # 4. Feature Selection
    # Note: This is computationally very intensive (esp. ReliefF)
    selected_features_indices = detector.feature_selection(
        train_features, train_labels, val_features, val_labels
    )

    # 5. Train Final Classifier
    detector.train_knn_classifier(train_features, train_labels, selected_features_indices)

    # 6. Evaluate on Test Set
    test_predictions, test_probabilities = detector.predict(
        test_features, selected_features_indices
    )

    test_accuracy = accuracy_score(test_labels, test_predictions)
    test_auc = roc_auc_score(test_labels, test_probabilities[:, 1])
    cm = confusion_matrix(test_labels, test_predictions)

    print("\n--- FINAL TEST RESULTS ---")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print(f"  Test AUC: {test_auc:.4f}")
    print(f"  Original features: {train_features.shape[1]}")
    print(f"  Selected features: {len(selected_features_indices)}")
    print(f"  Feature reduction: {(1 - len(selected_features_indices)/train_features.shape[1])*100:.2f}%")
    print("  Confusion Matrix:")
    print(cm)
    print("-------------------------")
