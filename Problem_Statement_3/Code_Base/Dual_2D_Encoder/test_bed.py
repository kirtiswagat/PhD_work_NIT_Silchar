
# =====================================================
# 1. Dual Modal Encoder
# =====================================================
class DualModalEncoder(Model):
    def __init__(self, input_shape=(224,224,3), feature_dim=512, dropout_rate=0.3):
        super().__init__()

        self.xray_encoder = EfficientNetB0(
            include_top=False, weights='imagenet',
            input_shape=input_shape, pooling='avg'
        )

        self.ct_encoder = EfficientNetB0(
            include_top=False, weights='imagenet',
            input_shape=input_shape, pooling='avg'
        )

        self.xray_feat = tf.keras.Sequential([
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(feature_dim, activation='relu')
        ])

        self.ct_feat = tf.keras.Sequential([
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(feature_dim, activation='relu')
        ])

        self.fusion = tf.keras.Sequential([
            layers.Concatenate(),
            layers.Dense(feature_dim * 2, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(feature_dim, activation='relu')
        ])

    def call(self, inputs, training=None):
        xray, ct = inputs
        xray_f = self.xray_feat(self.xray_encoder(xray, training=training))
        ct_f = self.ct_feat(self.ct_encoder(ct, training=training))
        fused = self.fusion([xray_f, ct_f])
        return fused


# =====================================================
# 2. Build Multi-Task Model
# =====================================================
def build_classifier(encoder, num_diseases, num_cancer_classes):
    xray_in = layers.Input(shape=(224,224,3))
    ct_in = layers.Input(shape=(224,224,3))

    features = encoder([xray_in, ct_in])

    disease_out = layers.Dense(
        num_diseases, activation='sigmoid', name='disease_out'
    )(features)

    cancer_out = layers.Dense(
        num_cancer_classes, activation='softmax', name='cancer_out'
    )(features)

    return Model([xray_in, ct_in], [disease_out, cancer_out])


# =====================================================
# 3. Image Loader
# =====================================================
def load_and_preprocess_image(path, image_size=(224,224)):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    return img / 255.0


# =====================================================
# 4. Dataset Creation
# =====================================================
def create_dataset_from_folder(base_dir, class_names, batch_size=32, multi_label=True, max_samples_per_class=None):
    all_paths, all_labels = [], []
    label_map = {c: i for i, c in enumerate(class_names)}

    print(f"Scanning files in {base_dir}...")
    # Use tqdm to show progress during file scanning
    for root, _, files in tqdm(os.walk(base_dir), desc=f"Scanning {base_dir}"):
        for f in files:
            if f.lower().endswith(('.jpg','.png','.jpeg')):
                full = os.path.join(root, f)
                rel = os.path.relpath(full, base_dir).split(os.sep)

                if multi_label:
                    vec = np.zeros(len(class_names))
                    has_relevant_label = False
                    for p in rel:
                        if p in label_map:
                            vec[label_map[p]] = 1
                            has_relevant_label = True
                    if not has_relevant_label: # If no relevant labels found, skip
                        continue
                else:
                    # For single-label classification, assume the first part of the path relative to base_dir is the class
                    # This logic might need adjustment based on specific folder structure for single-label
                    if rel and rel[0] in label_map:
                        vec = label_map[rel[0]]
                    else:
                        # Handle cases where the class cannot be determined or is not in label_map
                        continue # Skip this file if its class isn't found

                all_paths.append(full)
                all_labels.append(vec)
    print(f"Found {len(all_paths)} files in {base_dir}.")

    # --- Implement class-wise sampling if max_samples_per_class is provided ---
    if max_samples_per_class is not None:
        selected_paths_set = set()
        class_positive_counts = {i: 0 for i in range(len(class_names))}

        combined = list(zip(all_paths, all_labels))
        np.random.shuffle(combined)

        final_paths = []
        final_labels = []

        for path, label_vec in combined:
            add_file = False
            if multi_label:
                for i, val in enumerate(label_vec):
                    if val == 1 and class_positive_counts[i] < max_samples_per_class:
                        add_file = True
                        break
            else:
                # For single-label, if the class needs more samples
                if class_positive_counts[label_vec] < max_samples_per_class:
                    add_file = True

            if add_file and path not in selected_paths_set:
                final_paths.append(path)
                final_labels.append(label_vec)
                selected_paths_set.add(path)

                if multi_label:
                    for i, val in enumerate(label_vec):
                        if val == 1:
                            class_positive_counts[i] += 1
                else:
                    class_positive_counts[label_vec] += 1

            # Check if all classes have reached their limit (or if all files have been processed)
            if all(count >= max_samples_per_class for count in class_positive_counts.values()) or len(selected_paths_set) == len(all_paths):
                break

        paths = final_paths
        labels = final_labels
        print(f"After sampling, using {len(paths)} files for {base_dir}.")
        for i, class_name in enumerate(class_names):
            print(f"  Class '{class_name}': {class_positive_counts[i]} positive samples.")


    def gen():
        for p,l in tqdm(zip(paths, labels), total=len(paths), desc="Loading X-ray images"):
            yield load_and_preprocess_image(p), l

    output_signature_label = tf.TensorSpec((len(class_names),), tf.float32) if multi_label else tf.TensorSpec((), tf.int32)

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec((224,224,3), tf.float32),
            output_signature_label
        )
    )

    # Shuffle the entire dataset once after collecting all paths if there are enough files to make it efficient
    # Otherwise, shuffle can be applied per batch later.
    # For large datasets, shuffle might not be ideal here if it needs to load all into memory for shuffling.
    # However, from_generator is already handling data loading lazily.
    # It's better to shuffle a smaller buffer or rely on batch-level shuffling if needed.
    return ds.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def create_ct_dataset(filepaths, labels, batch_size=32):
    print(f"Preparing CT dataset from {len(filepaths)} files...")
    def gen():
        for p,l in tqdm(zip(filepaths, labels), total=len(filepaths), desc="Loading CT images"):
            yield load_and_preprocess_image(p), l

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec((224,224,3), tf.float32),
            tf.TensorSpec((), tf.int32)
        )
    )

    return ds.shuffle(len(filepaths)).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def create_paired_dataset(xray_ds, ct_ds):
    return tf.data.Dataset.zip((xray_ds, ct_ds)).map(
        lambda x, y: ([x[0], y[0]], [x[1], y[1]])
    )


# =====================================================
# 5. CLASS-BALANCED LOSS (ITEM-3)
# =====================================================
def compute_class_weights(y):
    pos = np.sum(y, axis=0)
    neg = y.shape[0] - pos
    return neg / (pos + 1e-6)


class WeightedBinaryCE(tf.keras.losses.Loss):
    def __init__(self, weights):
        super().__init__()
        self.w = tf.constant(weights, tf.float32)

    def call(self, y_true, y_pred):
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        weight = y_true * self.w + (1 - y_true)
        return tf.reduce_mean(bce * weight)


# =====================================================
# 6. TRAINING
# =====================================================
def train_model(model, train_ds, val_ds, loss_fn):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss={
            'disease_out': loss_fn,
            'cancer_out': 'sparse_categorical_crossentropy'
        },
        metrics={
            'disease_out': ['accuracy', tf.keras.metrics.AUC()],
            'cancer_out': ['accuracy']
        }
    )

    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ]
    )


# =====================================================
# 7. EVALUATION (ITEM-4 & 5)
# =====================================================
def evaluate_disease(model, dataset, class_names):
    y_true, y_scores = [], []

    for x,y in tqdm(dataset, desc="Evaluating disease predictions"):
        p = model.predict(x)[0]
        y_true.append(y[0].numpy())
        y_scores.append(p)

    y_true = np.vstack(y_true)
    y_scores = np.vstack(y_scores)

    print("\nF1 Scores")
    print("Micro:", f1_score(y_true, y_scores>=0.5, average='micro'))
    print("Macro:", f1_score(y_true, y_scores>=0.5, average='macro'))
    print("Weighted:", f1_score(y_true, y_scores>=0.5, average='weighted'))

    print("\nPer-Class AUC")
    for i,c in enumerate(class_names):
        print(c, roc_auc_score(y_true[:,i], y_scores[:,i]))

    print("\nOptimal Thresholds")
    thresholds = {}
    for i,c in enumerate(class_names):
        best_f1, best_t = 0, 0.5
        for t in np.arange(0.1,0.9,0.05):
            f1 = f1_score(y_true[:,i], (y_scores[:,i]>=t).astype(int))
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = best_t
        print(f"{c}: {best_t:.2f}")
    return thresholds


# =====================================================
# 8. MAIN
# =====================================================
def main():
    xray_classes = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Pleural_Effusion',
    ]
    # 'Enlarged_Cardiomediastinum','Fracture','Lung_Lesion','Lung_Opacity',
    # 'No_Finding','Pleural_Other','Pneumonia','Pneumothorax','Support_Devices'
    ct_classes = ['benign', 'malignant', 'normal']

    xray_train = '/content/drive/MyDrive/Kaggel_direct_download/CheXpert-v1.0-small/train'
    xray_test = '/content/drive/MyDrive/Kaggel_direct_download/CheXpert-v1.0-small/valid'
    ct_dir = '/content/drive/MyDrive/Kaggel_direct_download/Cancel_Dataset'

    # Limit X-ray dataset to at least 100 samples per class
    xray_train_ds = create_dataset_from_folder(xray_train, xray_classes, max_samples_per_class=100)
    xray_val_ds = create_dataset_from_folder(xray_test, xray_classes, max_samples_per_class=100)

    # Extract X-ray labels for class weights
    labels = []
    for _, y in xray_train_ds:
        labels.append(y.numpy())
    weights = compute_class_weights(np.vstack(labels))

    ct_files_all, ct_labels_all = [], []
    for i, c in enumerate(ct_classes):
        class_specific_files = [os.path.join(ct_dir, c, f) for f in os.listdir(os.path.join(ct_dir, c)) if
                                f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        # Randomly sample up to 500 files for this class (which is >= 100)
        if len(class_specific_files) > 500:
            sampled_files = np.random.choice(class_specific_files, 500, replace=False)
        else:
            sampled_files = class_specific_files

        ct_files_all.extend(sampled_files)
        ct_labels_all.extend([i] * len(sampled_files))

    ct_files, ct_labels = np.array(ct_files_all), np.array(ct_labels_all)
    print(f"Total CT files after sampling: {len(ct_files)}")

    tr_f, va_f, tr_l, va_l = train_test_split(
        ct_files, ct_labels, stratify=ct_labels, test_size=0.3
    )

    ct_train_ds = create_ct_dataset(tr_f, tr_l)
    ct_val_ds = create_ct_dataset(va_f, va_l)

    train_ds = create_paired_dataset(xray_train_ds, ct_train_ds)
    val_ds = create_paired_dataset(xray_val_ds, ct_val_ds)

    model = build_classifier(
        DualModalEncoder(),
        len(xray_classes),
        len(ct_classes)
    )

    history = train_model(
        model,
        train_ds,
        val_ds,
        WeightedBinaryCE(weights)
    )

    evaluate_disease(model, val_ds, xray_classes)