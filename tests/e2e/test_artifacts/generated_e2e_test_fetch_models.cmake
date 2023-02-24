iree_fetch_artifact(
  NAME
    "model-DeepLabV3_fp32_fp32_"
  SOURCE_URL
    "https://storage.googleapis.com/iree-model-artifacts/deeplabv3.tflite"
  OUTPUT
    "${ROOT_ARTIFACTS_DIR}/model_DeepLabV3_fp32[fp32].tflite"
  UNPACK
)

iree_fetch_artifact(
  NAME
    "model-EfficientNet_int8_int8_"
  SOURCE_URL
    "https://storage.googleapis.com/iree-model-artifacts/efficientnet_lite0_int8_2.tflite"
  OUTPUT
    "${ROOT_ARTIFACTS_DIR}/model_EfficientNet_int8[int8].tflite"
  UNPACK
)

iree_fetch_artifact(
  NAME
    "model-MobileBertSquad_fp16_fp16_"
  SOURCE_URL
    "https://storage.googleapis.com/iree-model-artifacts/mobilebertsquad.tflite"
  OUTPUT
    "${ROOT_ARTIFACTS_DIR}/model_MobileBertSquad_fp16[fp16].tflite"
  UNPACK
)

iree_fetch_artifact(
  NAME
    "model-MobileBertSquad_fp32_fp32_"
  SOURCE_URL
    "https://storage.googleapis.com/iree-model-artifacts/mobilebert-baseline-tf2-float.tflite"
  OUTPUT
    "${ROOT_ARTIFACTS_DIR}/model_MobileBertSquad_fp32[fp32].tflite"
  UNPACK
)

iree_fetch_artifact(
  NAME
    "model-MobileBertSquad_int8_int8_"
  SOURCE_URL
    "https://storage.googleapis.com/iree-model-artifacts/mobilebert-baseline-tf2-quant.tflite"
  OUTPUT
    "${ROOT_ARTIFACTS_DIR}/model_MobileBertSquad_int8[int8].tflite"
  UNPACK
)

iree_fetch_artifact(
  NAME
    "model-MobileNetV1_fp32_fp32_imagenet_"
  SOURCE_URL
    "https://storage.googleapis.com/iree-model-artifacts/mobilenet_v1_224_1.0_float.tflite"
  OUTPUT
    "${ROOT_ARTIFACTS_DIR}/model_MobileNetV1_fp32[fp32,imagenet].0_float.tflite"
  UNPACK
)

iree_fetch_artifact(
  NAME
    "model-MobileNetV2_fp32_fp32_imagenet_"
  SOURCE_URL
    "https://storage.googleapis.com/iree-model-artifacts/mobilenet_v2_1.0_224.tflite"
  OUTPUT
    "${ROOT_ARTIFACTS_DIR}/model_MobileNetV2_fp32[fp32,imagenet].0_224.tflite"
  UNPACK
)

iree_fetch_artifact(
  NAME
    "model-MobileNetV3Small_fp32_fp32_imagenet_"
  SOURCE_URL
    "https://storage.googleapis.com/iree-model-artifacts/MobileNetV3SmallStaticBatch.tflite"
  OUTPUT
    "${ROOT_ARTIFACTS_DIR}/model_MobileNetV3Small_fp32[fp32,imagenet].tflite"
  UNPACK
)

iree_fetch_artifact(
  NAME
    "model-MobileSSD_fp32_fp32_"
  SOURCE_URL
    "https://storage.googleapis.com/iree-model-artifacts/mobile_ssd_v2_float_coco.tflite"
  OUTPUT
    "${ROOT_ARTIFACTS_DIR}/model_MobileSSD_fp32[fp32].tflite"
  UNPACK
)

iree_fetch_artifact(
  NAME
    "model-PersonDetect_int8_int8_"
  SOURCE_URL
    "https://storage.googleapis.com/iree-model-artifacts/person_detect.tflite"
  OUTPUT
    "${ROOT_ARTIFACTS_DIR}/model_PersonDetect_int8[int8].tflite"
  UNPACK
)

iree_fetch_artifact(
  NAME
    "model-PoseNet_fp32_fp32_"
  SOURCE_URL
    "https://storage.googleapis.com/iree-model-artifacts/posenet.tflite"
  OUTPUT
    "${ROOT_ARTIFACTS_DIR}/model_PoseNet_fp32[fp32].tflite"
  UNPACK
)

iree_fetch_artifact(
  NAME
    "model-BertForMaskedLMTF_fp32_seqlen512_tensorflow_"
  SOURCE_URL
    "https://storage.googleapis.com/iree-model-artifacts/bert-for-masked-lm-seq512-tf-model.tar.gz"
  OUTPUT
    "${ROOT_ARTIFACTS_DIR}/model_BertForMaskedLMTF[fp32,seqlen512,tensorflow]"
  UNPACK
)

iree_fetch_artifact(
  NAME
    "model-BertLargeTF_fp32_seqlen384_tensorflow_"
  SOURCE_URL
    "https://storage.googleapis.com/iree-model-artifacts/bert-large-seq384-tf-model.tar.gz"
  OUTPUT
    "${ROOT_ARTIFACTS_DIR}/model_BertLargeTF[fp32,seqlen384,tensorflow]"
  UNPACK
)

iree_fetch_artifact(
  NAME
    "model-EfficientNetV2STF_fp32_cnn_tensorflow_"
  SOURCE_URL
    "https://storage.googleapis.com/iree-model-artifacts/efficientnet-v2-s-tf-model.tar.gz"
  OUTPUT
    "${ROOT_ARTIFACTS_DIR}/model_EfficientNetV2STF[fp32,cnn,tensorflow]"
  UNPACK
)

iree_fetch_artifact(
  NAME
    "model-MiniLML12H384Uncased_int32_seqlen128_"
  SOURCE_URL
    "https://storage.googleapis.com/iree-model-artifacts/minilm-l12-h384-uncased-seqlen128-tf-model.tar.gz"
  OUTPUT
    "${ROOT_ARTIFACTS_DIR}/model_MiniLML12H384Uncased[int32,seqlen128]"
  UNPACK
)

iree_fetch_artifact(
  NAME
    "model-Resnet50TF_fp32_"
  SOURCE_URL
    "https://storage.googleapis.com/iree-model-artifacts/resnet50-tf-model.tar.gz"
  OUTPUT
    "${ROOT_ARTIFACTS_DIR}/model_Resnet50TF[fp32]"
  UNPACK
)
