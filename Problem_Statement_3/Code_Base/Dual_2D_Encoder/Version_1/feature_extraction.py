class DualModalEncoder(Model):
    def __init__(self, input_shape=(224,224,3), feature_dim=512, dropout_rate=0.3):
        super().__init__()
        # X-ray Encoder
        self.xray_encoder = EfficientNetB0(include_top=False,
                                          weights='imagenet',
                                          input_shape=input_shape,
                                          pooling='avg')
        self.xray_encoder.trainable = True
        self.xray_feat_extractor = keras.Sequential([
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(feature_dim, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate/2),
        ])
        # CT Encoder
        self.ct_encoder = EfficientNetB0(include_top=False,
                                         weights='imagenet',
                                         input_shape=input_shape,
                                         pooling='avg')
        self.ct_encoder.trainable = True
        self.ct_feat_extractor = keras.Sequential([
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(feature_dim, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate/2),
        ])
        # Fusion Layers
        self.fusion = keras.Sequential([
            layers.Concatenate(),
            layers.Dense(feature_dim*2, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(feature_dim, activation='relu'),
            layers.BatchNormalization(),
        ])

    def call(self, inputs, training=None):
        xray_input, ct_input = inputs
        xray_features = self.xray_encoder(xray_input, training=training)
        xray_features = self.xray_feat_extractor(xray_features, training=training)
        ct_features = self.ct_encoder(ct_input, training=training)
        ct_features = self.ct_feat_extractor(ct_features, training=training)
        concatenated = tf.concat([xray_features, ct_features], axis=-1)
        fused = self.fusion(concatenated, training=training)
        return fused
def build_classifier(dual_encoder, num_diseases, num_cancer_classes):
    # Output both tasks: lung diseases and cancer type
    inputs_xray = layers.Input(shape=(224,224,3))
    inputs_ct = layers.Input(shape=(224,224,3))
    features = dual_encoder([inputs_xray, inputs_ct])
    
    # Disease prediction head (multi-label)
    disease_output = layers.Dense(num_diseases, activation='sigmoid', name='disease_out')(features)
    
    # Cancer classification (single/multi-class)
    cancer_output = layers.Dense(num_cancer_classes, activation='softmax', name='cancer_out')(features)
    
    model = Model(inputs=[inputs_xray, inputs_ct], outputs=[disease_output, cancer_output])
    return model
