import torch
import timm
import torch
from transformers import ViTImageProcessor, VivitModel

# Load model and image processor for selected model

# model_name = "nateraw/vit-age-classifier"
# output_name = "vit_age_classifier.onnx"

model_name = "weights"
#model_name = "cledoux42/Age_Classify_v001"
output_name = "vit_model.onnx"

#model = VivitModel.from_pretrained(model_name)
#feature_extractor = ViTImageProcessor.from_pretrained(model_name)


model = timm.create_model('xception', pretrained=False)

state_dict = torch.load('weights/xception-b5690688.pth', map_location='cpu')

# The state_dict keys may have a 'module.' prefix if the model was trained with
# DataParallel. You may need to remove this prefix for a successful load.
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k # remove `module.` prefix
    new_state_dict[name] = v

# Load the modified state_dict into the model
model.load_state_dict(new_state_dict)


# Set model to evaluation mode
model.eval()

# Get correct image size
image_size = feature_extractor.size["height"]  # Default: 224
dummy_image = torch.randn(1, 3, image_size, image_size)

# Convert to ONNX with updated opset version (14+)
onnx_path = output_name
torch.onnx.export(
    model,
    (dummy_image,),
    onnx_path,
    input_names=["pixel_values"],
    output_names=["logits"],
    dynamic_axes={"pixel_values": {0: "batch_size"}, "logits": {0: "batch_size"}},
    opset_version=14,
)

print(f"Model successfully converted to ONNX and saved at: {onnx_path}")


# The onnx file is too large to upload to github, you must generate it yourself by running this script