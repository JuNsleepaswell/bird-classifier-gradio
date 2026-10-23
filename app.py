
# app.py -- Gradio demo with a GradCAM++ explanation, generated from notebooks/main.ipynb (Phase 7.7).

# no torch at runtime, GradCAM++ runs in numpy using weight[c,k]/(H*W) as the layer4 gradient.

import json
import os

import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "model.onnx")
CLASSIFIER_WEIGHTS_PATH = os.path.join(APP_DIR, "classifier_weights.npz")
CLASS_NAMES_PATH = os.path.join(APP_DIR, "class_names.json")
EXAMPLES_DIR = os.path.join(APP_DIR, "examples")

IMG_SIZE = 224
EVAL_RESIZE = 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
LOW_CONFIDENCE_THRESHOLD = 0.30
CAM_IMAGE_WEIGHT = 0.5  # how much of the original image shows through the heatmap

with open(CLASS_NAMES_PATH) as f:
    CLASS_NAMES = json.load(f)

_classifier = np.load(CLASSIFIER_WEIGHTS_PATH)
CLASSIFIER_WEIGHT = _classifier["weight"]  # (200, 2048)
CLASSIFIER_BIAS = _classifier["bias"]  # (200,)

SESSION = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])


def _resize_and_crop(image):
    # resize short side to 256 (truncated) and crop offsets rounded, same formula as the streamlit app
    image = image.convert("RGB")
    w, h = image.size
    if w <= h:
        new_w, new_h = EVAL_RESIZE, int(EVAL_RESIZE * h / w)
    else:
        new_h, new_w = EVAL_RESIZE, int(EVAL_RESIZE * w / h)
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    crop_left = int(round((new_w - IMG_SIZE) / 2.0))
    crop_top = int(round((new_h - IMG_SIZE) / 2.0))
    return resized.crop((crop_left, crop_top, crop_left + IMG_SIZE, crop_top + IMG_SIZE))


def preprocess(image):
    # returns the normalised NCHW tensor plus the cropped 224x224 image used as the GradCAM overlay base
    cropped = _resize_and_crop(image)
    arr = np.asarray(cropped).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    tensor = arr.transpose(2, 0, 1)[None].astype(np.float32)
    return tensor, cropped


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def run_model(tensor):
    logits, layer4_features = SESSION.run(["logits", "layer4_features"], {"input": tensor})
    return logits[0], layer4_features[0]  # (200,), (2048, H, W)


def gradcam_plusplus(layer4_features, class_weight_row):
    # pytorch_grad_cam's GradCAMPlusPlus formula, gradient is constant per channel under GAP+Linear
    C, H, W = layer4_features.shape
    grads = np.broadcast_to((class_weight_row / (H * W))[:, None, None], (C, H, W)).astype(np.float32)
    grads_power_2 = grads ** 2
    grads_power_3 = grads_power_2 * grads
    sum_activations = layer4_features.sum(axis=(1, 2), keepdims=True)
    eps = 1e-6
    aij = grads_power_2 / (2 * grads_power_2 + sum_activations * grads_power_3 + eps)
    aij = np.where(grads != 0, aij, 0)
    weights = np.sum(np.maximum(grads, 0) * aij, axis=(1, 2))  # (C,)

    cam = (weights[:, None, None] * layer4_features).sum(axis=0)  # (H, W)
    cam = np.maximum(cam, 0)
    cam = cam - cam.min()
    cam = cam / (1e-7 + cam.max())

    # pytorch_grad_cam normalises then resizes to input size, here via PIL BILINEAR instead of cv2
    cam_resized = np.asarray(
        Image.fromarray(cam.astype(np.float32), mode="F").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR),
        dtype=np.float32,
    )
    # matches aggregate_multi_layers' second maximum(...,0) + re-normalise pass
    cam_resized = np.maximum(cam_resized, 0)
    cam_resized = cam_resized - cam_resized.min()
    cam_resized = cam_resized / (1e-7 + cam_resized.max())
    return cam_resized  # (IMG_SIZE, IMG_SIZE) float32 in [0, 1]


def jet_colormap(x):
    # small jet-like colormap, piecewise-linear approximation, no matplotlib
    x = np.clip(x, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)  # (H, W, 3) float in [0, 1]


def colorize_and_blend(cropped_image, cam_map, image_weight=CAM_IMAGE_WEIGHT):
    base = np.asarray(cropped_image).astype(np.float32) / 255.0
    heatmap = jet_colormap(cam_map)
    blended = image_weight * base + (1 - image_weight) * heatmap
    blended = np.clip(blended, 0.0, 1.0)
    return Image.fromarray((blended * 255).astype(np.uint8))


def predict(image):
    # PIL image in, {display_name: confidence} out, for gr.Label(num_top_classes=5)
    tensor, _ = preprocess(image)
    logits, _ = run_model(tensor)
    probs = softmax(logits)
    top5_idx = np.argsort(probs)[::-1][:5]
    return {CLASS_NAMES[i].replace("_", " "): float(probs[i]) for i in top5_idx}


def gradcam_overlay(image, class_idx=None):
    # PIL image in, PIL image out, defaults to the model's own top-1 class if none is given
    tensor, cropped = preprocess(image)
    logits, layer4_features = run_model(tensor)
    if class_idx is None:
        class_idx = int(np.argmax(logits))
    cam_map = gradcam_plusplus(layer4_features, CLASSIFIER_WEIGHT[class_idx])
    return colorize_and_blend(cropped, cam_map)


def classify(image):
    # single entry point the Gradio UI calls, does one forward pass for all three outputs
    if image is None:
        return {}, None, gr.update(visible=False)

    tensor, cropped = preprocess(image)
    logits, layer4_features = run_model(tensor)
    probs = softmax(logits)
    top5_idx = np.argsort(probs)[::-1][:5]
    label_dict = {CLASS_NAMES[i].replace("_", " "): float(probs[i]) for i in top5_idx}

    top1_idx = int(top5_idx[0])
    top1_prob = float(probs[top1_idx])
    cam_map = gradcam_plusplus(layer4_features, CLASSIFIER_WEIGHT[top1_idx])
    overlay = colorize_and_blend(cropped, cam_map)

    if top1_prob < LOW_CONFIDENCE_THRESHOLD:
        warning = gr.update(
            value=(
                "**Low confidence** (top prediction under 30%). This photo might "
                "not be one of the 200 CUB-200 species, or the bird is hard to make out."
            ),
            visible=True,
        )
    else:
        warning = gr.update(visible=False)

    return label_dict, overlay, warning


with gr.Blocks(title="CUB-200 Bird Species Classifier") as demo:
    gr.Markdown("# CUB-200 Bird Species Classifier")
    gr.Markdown(
        "Fine-tuned ResNet-50 on 200 CUB-200 bird species. Upload a photo or pick "
        "one of the examples below to see the top-5 guesses and a GradCAM++ "
        "heatmap showing which part of the photo drove the top prediction."
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload a bird photo")
        with gr.Column():
            label_output = gr.Label(num_top_classes=5, label="Top-5 predicted species")
            cam_output = gr.Image(label="GradCAM++ (top-1 class)")
            warning_output = gr.Markdown(visible=False)

    gr.Examples(
        examples=[
            os.path.join(EXAMPLES_DIR, "example_1.jpg"),
            os.path.join(EXAMPLES_DIR, "example_2.jpg"),
            os.path.join(EXAMPLES_DIR, "example_3.jpg"),
        ],
        inputs=image_input,
        outputs=[label_output, cam_output, warning_output],
        fn=classify,
        cache_examples=False,
        label="Example photos",
    )

    image_input.change(fn=classify, inputs=image_input, outputs=[label_output, cam_output, warning_output])

    gr.Markdown("---")
    gr.Markdown(
        "Model: resnet50_finetune_aug_strong_testsel (ResNet-50, fine-tuned). "
        "85.7% Top-1 on the CUB-200 test set (test-set selection protocol; "
        "83.5% under the held-out validation-split protocol)."
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        theme=gr.themes.Soft(),
    )
