# Fetch the model from "https://storage.googleapis.com/iree-model-artifacts/deeplabv3.tflite"
iree_fetch_artifact(
  NAME
    "model-c36c63b0-220a-4d78-8ade-c45ce47d89d3"
  SOURCE_URL
    "https://storage.googleapis.com/iree-model-artifacts/deeplabv3.tflite"
  OUTPUT
    "${ROOT_ARTIFACTS_DIR}/models/c36c63b0-220a-4d78-8ade-c45ce47d89d3_DeepLabV3_fp32.tflite"
  UNPACK
)

# Fetch the model from "https://storage.googleapis.com/iree-model-artifacts/mobile_ssd_v2_float_coco.tflite"
iree_fetch_artifact(
  NAME
    "model-0e466f69-91d6-4e50-b62b-a82b6213a231"
  SOURCE_URL
    "https://storage.googleapis.com/iree-model-artifacts/mobile_ssd_v2_float_coco.tflite"
  OUTPUT
    "${ROOT_ARTIFACTS_DIR}/models/0e466f69-91d6-4e50-b62b-a82b6213a231_MobileSSD_fp32.tflite"
  UNPACK
)

# Fetch the model from "https://storage.googleapis.com/iree-model-artifacts/posenet.tflite"
iree_fetch_artifact(
  NAME
    "model-5afc3014-d29d-4e88-a840-fbaf678acf2b"
  SOURCE_URL
    "https://storage.googleapis.com/iree-model-artifacts/posenet.tflite"
  OUTPUT
    "${ROOT_ARTIFACTS_DIR}/models/5afc3014-d29d-4e88-a840-fbaf678acf2b_PoseNet_fp32.tflite"
  UNPACK
)

# Fetch the model from "https://storage.googleapis.com/iree-model-artifacts/mobilebert-baseline-tf2-float.tflite"
iree_fetch_artifact(
  NAME
    "model-cc69d69f-6d1f-4a1a-a31e-e021888d0d28"
  SOURCE_URL
    "https://storage.googleapis.com/iree-model-artifacts/mobilebert-baseline-tf2-float.tflite"
  OUTPUT
    "${ROOT_ARTIFACTS_DIR}/models/cc69d69f-6d1f-4a1a-a31e-e021888d0d28_MobileBertSquad_fp32.tflite"
  UNPACK
)

# Fetch the model from "https://storage.googleapis.com/iree-model-artifacts/mobilebert-baseline-tf2-quant.tflite"
iree_fetch_artifact(
  NAME
    "model-e3997104-a3d2-46b4-9fbf-39069906d123"
  SOURCE_URL
    "https://storage.googleapis.com/iree-model-artifacts/mobilebert-baseline-tf2-quant.tflite"
  OUTPUT
    "${ROOT_ARTIFACTS_DIR}/models/e3997104-a3d2-46b4-9fbf-39069906d123_MobileBertSquad_int8.tflite"
  UNPACK
)

# Fetch the model from "https://storage.googleapis.com/iree-model-artifacts/mobilebertsquad.tflite"
iree_fetch_artifact(
  NAME
    "model-73a0402e-271b-4aa8-a6a5-ac05839ca569"
  SOURCE_URL
    "https://storage.googleapis.com/iree-model-artifacts/mobilebertsquad.tflite"
  OUTPUT
    "${ROOT_ARTIFACTS_DIR}/models/73a0402e-271b-4aa8-a6a5-ac05839ca569_MobileBertSquad_fp16.tflite"
  UNPACK
)

# Fetch the model from "https://storage.googleapis.com/iree-model-artifacts/mobilenet_v1_224_1.0_float.tflite"
iree_fetch_artifact(
  NAME
    "model-78eab9e5-9ff1-4769-9b55-933c81cc9a0f"
  SOURCE_URL
    "https://storage.googleapis.com/iree-model-artifacts/mobilenet_v1_224_1.0_float.tflite"
  OUTPUT
    "${ROOT_ARTIFACTS_DIR}/models/78eab9e5-9ff1-4769-9b55-933c81cc9a0f_MobileNetV1_fp32.0_float.tflite"
  UNPACK
)

# Fetch the model from "https://storage.googleapis.com/iree-model-artifacts/mobilenet_v2_1.0_224.tflite"
iree_fetch_artifact(
  NAME
    "model-7d45f8e5-bb5e-48d0-928d-8f125104578f"
  SOURCE_URL
    "https://storage.googleapis.com/iree-model-artifacts/mobilenet_v2_1.0_224.tflite"
  OUTPUT
    "${ROOT_ARTIFACTS_DIR}/models/7d45f8e5-bb5e-48d0-928d-8f125104578f_MobileNetV2_fp32.0_224.tflite"
  UNPACK
)

# Fetch the model from "https://storage.googleapis.com/iree-model-artifacts/MobileNetV3SmallStaticBatch.tflite"
iree_fetch_artifact(
  NAME
    "model-58855e40-eba9-4a71-b878-6b35e3460244"
  SOURCE_URL
    "https://storage.googleapis.com/iree-model-artifacts/MobileNetV3SmallStaticBatch.tflite"
  OUTPUT
    "${ROOT_ARTIFACTS_DIR}/models/58855e40-eba9-4a71-b878-6b35e3460244_MobileNetV3Small_fp32.tflite"
  UNPACK
)

# Fetch the model from "https://storage.googleapis.com/iree-model-artifacts/person_detect.tflite"
iree_fetch_artifact(
  NAME
    "model-bc1338be-e3df-44fd-82e4-40ba9560a073"
  SOURCE_URL
    "https://storage.googleapis.com/iree-model-artifacts/person_detect.tflite"
  OUTPUT
    "${ROOT_ARTIFACTS_DIR}/models/bc1338be-e3df-44fd-82e4-40ba9560a073_PersonDetect_int8.tflite"
  UNPACK
)

# Fetch the model from "https://storage.googleapis.com/iree-model-artifacts/efficientnet_lite0_int8_2.tflite"
iree_fetch_artifact(
  NAME
    "model-4a6f545e-1b4e-41a5-9236-792aa578184b"
  SOURCE_URL
    "https://storage.googleapis.com/iree-model-artifacts/efficientnet_lite0_int8_2.tflite"
  OUTPUT
    "${ROOT_ARTIFACTS_DIR}/models/4a6f545e-1b4e-41a5-9236-792aa578184b_EfficientNet_int8.tflite"
  UNPACK
)

# Fetch the model from "https://storage.googleapis.com/iree-model-artifacts/minilm-l12-h384-uncased-seqlen128-tf-model.tar.gz"
iree_fetch_artifact(
  NAME
    "model-ecf5c970-ee97-49f0-a4ed-df1f34e9d493"
  SOURCE_URL
    "https://storage.googleapis.com/iree-model-artifacts/minilm-l12-h384-uncased-seqlen128-tf-model.tar.gz"
  OUTPUT
    "${ROOT_ARTIFACTS_DIR}/models/ecf5c970-ee97-49f0-a4ed-df1f34e9d493_MiniLML12H384Uncased"
  UNPACK
)
