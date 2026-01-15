from tensorflow.keras.applications import (
    Xception,
    ResNet50, ResNet50V2,
    ResNet101, ResNet101V2,
    ResNet152, ResNet152V2,
    InceptionV3, InceptionResNetV2,
    MobileNet, MobileNetV2,
    DenseNet121, DenseNet169, DenseNet201,
    NASNetMobile, NASNetLarge,
    EfficientNetB0, EfficientNetB1, EfficientNetB2,
    EfficientNetB3, EfficientNetB4, EfficientNetB5,
    EfficientNetB6, EfficientNetB7,
    EfficientNetV2B0, EfficientNetV2B1,
    EfficientNetV2B2, EfficientNetV2B3,
    EfficientNetV2S, EfficientNetV2M, EfficientNetV2L,
    ConvNeXtTiny, ConvNeXtSmall,
    ConvNeXtBase, ConvNeXtLarge, ConvNeXtXLarge
)


def get_backbone(backbone_name, input_shape):
    name = backbone_name.lower()

    if name == "xception":
        return Xception(include_top=False, weights="imagenet", pooling="avg", input_shape=input_shape)

    elif name == "vgg16":
        return VGG16(include_top=False, weights="imagenet", pooling="avg", input_shape=input_shape)

    elif name == "vgg19":
        return VGG19(include_top=False, weights="imagenet", pooling="avg", input_shape=input_shape)

    elif name == "resnet50":
        return ResNet50(include_top=False, weights="imagenet", pooling="avg", input_shape=input_shape)

    elif name == "resnet50v2":
        return ResNet50V2(include_top=False, weights="imagenet", pooling="avg", input_shape=input_shape)

    elif name == "resnet101":
        return ResNet101(include_top=False, weights="imagenet", pooling="avg", input_shape=input_shape)

    elif name == "resnet101v2":
        return ResNet101V2(include_top=False, weights="imagenet", pooling="avg", input_shape=input_shape)

    elif name == "resnet152":
        return ResNet152(include_top=False, weights="imagenet", pooling="avg", input_shape=input_shape)

    elif name == "resnet152v2":
        return ResNet152V2(include_top=False, weights="imagenet", pooling="avg", input_shape=input_shape)

    elif name == "inceptionv3":
        return InceptionV3(include_top=False, weights="imagenet", pooling="avg", input_shape=input_shape)

    elif name == "inceptionresnetv2":
        return InceptionResNetV2(include_top=False, weights="imagenet", pooling="avg", input_shape=input_shape)

    elif name == "mobilenet":
        return MobileNet(include_top=False, weights="imagenet", pooling="avg", input_shape=input_shape)

    elif name == "mobilenetv2":
        return MobileNetV2(include_top=False, weights="imagenet", pooling="avg", input_shape=input_shape)

    elif name == "densenet121":
        return DenseNet121(include_top=False, weights="imagenet", pooling="avg", input_shape=input_shape)

    elif name == "densenet169":
        return DenseNet169(include_top=False, weights="imagenet", pooling="avg", input_shape=input_shape)

    elif name == "densenet201":
        return DenseNet201(include_top=False, weights="imagenet", pooling="avg", input_shape=input_shape)

    elif name == "nasnetmobile":
        return NASNetMobile(include_top=False, weights="imagenet", pooling="avg", input_shape=input_shape)

    elif name == "nasnetlarge":
        return NASNetLarge(include_top=False, weights="imagenet", pooling="avg", input_shape=input_shape)

    elif name == "efficientnetb0":
        return EfficientNetB0(include_top=False, weights="imagenet", pooling="avg", input_shape=input_shape)

    elif name == "efficientnetv2s":
        return EfficientNetV2S(include_top=False, weights="imagenet", pooling="avg", input_shape=input_shape)

    elif name == "convnextbase":
        return ConvNeXtBase(include_top=False, weights="imagenet", pooling="avg", input_shape=input_shape)

    elif name == "vit":
        inputs = layers.Input(shape=input_shape)
        x = layers.Conv2D(64, 16, strides=16)(inputs)
        x = layers.Reshape((-1, 64))(x)
        x = LayerNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)
        return Model(inputs, x, name="ViT_Backbone")

    else:
        raise ValueError(f"❌ Unknown backbone: {backbone_name}")
