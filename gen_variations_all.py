import argparse
import os

import numpy as np
import json
import torch
import torchvision
from PIL import Image
import litellm

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    build_sam,
    build_sam_hq,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Recognize Anything Model & Tag2Text
from ram.models import ram
from ram import inference_ram
import torchvision.transforms as TS
from transformers import pipeline
from pattern.text.en import singularize

import torch
from diffusers import AutoPipelineForInpainting, AutoPipelineForImage2Image
from diffusers.utils import make_image_grid
from diffusers.utils import load_image as load_image_diffusers
from scipy.ndimage import label as scipy_label
import google.generativeai as genai
import re

import gc

def get_closest_size(width, height):
    # Defined aspect ratios and their corresponding dimensions
    aspect_ratios = {
        5 / 12: (640, 1536),
        4 / 7: (768, 1344),
        13 / 19: (832, 1216),
        7 / 9: (896, 1152),
        1 / 1: (1024, 1024),
        9 / 7: (1152, 896),
        19 / 13: (1216, 832),
        7 / 4: (1344, 768),
        12 / 5: (1536, 640),
    }

    # Calculate the aspect ratio of the input image
    input_aspect_ratio = width / height

    # Find the aspect ratio in the defined list that is closest to the input image's aspect ratio
    closest_aspect_ratio = min(aspect_ratios.keys(), key=lambda x: abs(x - input_aspect_ratio))

    # Return the dimensions corresponding to the closest aspect ratio
    return aspect_ratios[closest_aspect_ratio]


def load_image(image_path):
    # Load image
    image_pil = Image.open(image_path).convert("RGB")

    # Get the closest size based on aspect ratio
    new_width, new_height = get_closest_size(image_pil.width, image_pil.height)

    # Resize the image to the closest aspect ratio
    image_pil = image_pil.resize((new_width, new_height))

    # Define the transform (resize moved inside the get_closest_size function)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Transform the image
    image, _ = transform(image_pil, None)  # 3, h, w
    print("Shape of the image: ", image.shape)

    return image_pil, image


def check_tags_chinese(tags_chinese, pred_phrases, max_tokens=100, model="gpt-3.5-turbo"):
    object_list = [obj.split('(')[0] for obj in pred_phrases]
    object_num = []
    for obj in set(object_list):
        object_num.append(f'{object_list.count(obj)} {obj}')
    object_num = ', '.join(object_num)
    print(f"Correct object number: {object_num}")

    if openai_key:
        prompt = [
            {
                'role': 'system',
                'content': 'Revise the number in the tags_chinese if it is wrong. ' + \
                           f'tags_chinese: {tags_chinese}. ' + \
                           f'True object number: {object_num}. ' + \
                           'Only give the revised tags_chinese: '
            }
        ]
        response = litellm.completion(model=model, messages=prompt, temperature=0.6, max_tokens=max_tokens)
        reply = response['choices'][0]['message']['content']
        # sometimes return with "tags_chinese: xxx, xxx, xxx"
        tags_chinese = reply.split(':')[-1].strip()
    return tags_chinese


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, tags_chinese, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = {
        'tags_chinese': tags_chinese,
        'mask': [{
            'value': value,
            'label': 'background'
        }]
    }
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1]  # the last is ')'
        json_data['mask'].append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'label.json'), 'w') as f:
        json.dump(json_data, f)


caption_question = 'Can you please describe this image in up to two paragraphs? Please specify any objects within the image, backgrounds, scenery, interactions, and gestures or poses. If they are multiple of any object, please specify how many. Is there text in the image, and if so, what does it say? If there is any lighting in the image, can you identify where it is and what it looks like? What style is the image? If there are people or characters in the image, what emotions are they conveying? Please keep your descriptions factual and terse but complete. DO NOT add any unnecessary speculation about the things that are not part of the image such as "the image is inspiring to viewers" or "seeing this makes you feel joy". DO NOT add things such as "creates a unique and entertaining visual", as these descriptions are interpretations and not a part of the image itself. The description should be purely factual, with no subjective speculation. Make sure to include the style of the image, for example cartoon, photograph, 3d render etc. Start with the words "This image showcases":'

def convert_to_corners(boxes):
    # Assuming boxes are in (x_center, y_center, width, height) format
    xmin = boxes[:, 0] - (boxes[:, 2] / 2)
    ymin = boxes[:, 1] - (boxes[:, 3] / 2)
    xmax = boxes[:, 0] + (boxes[:, 2] / 2)
    ymax = boxes[:, 1] + (boxes[:, 3] / 2)
    return torch.stack([xmin, ymin, xmax, ymax], dim=1)

# Example conversion (adjust with actual tensor manipulation as needed)


def get_caption_segmentation(image_pil, image, transform):
    raw_image = image_pil.resize(
        (384, 384))
    raw_image = transform(raw_image).unsqueeze(0).to(device)

    res = inference_ram(raw_image, ram_model)

    tags_list = res[0].split(" | ")
    print("tags: ", tags_list)


    # this is to get the caption about the image.
    question = caption_question
    print(question)
    print("")
    original_caption = vision_model.generate_content([question, image_pil])
    original_caption.resolve()
    print(original_caption.text)

    print("")
    print("")
    print("#" * 100)


    # this is to extract all the objects that might be in this image.
    tags = res[0].replace(' |', ',')
    question = f"The description of an image is given below. \n '{original_caption.text}' \n and " \
               f"some more information about image is based on tags which are {tags}. " \
               f"Find all the objects even small that may be in this image based on this information separated by ,. " \
               f"Also do remove duplicates, verbs and adjectives. "
    print(question)

    response = text_model.generate_content(question, generation_config=genai.types.GenerationConfig(temperature=0.0))
    response.resolve()
    print(response.text)

    # objects_list = response.text.split(", ")
    objects_list = response.text
    objects_list = objects_list.replace(",", "")
    objects_list = objects_list.replace("\n", " ").replace("-", " ").replace(",", " ")

    pattern = "[^a-zA-Z0-9]"
    objects_list = re.sub(pattern, " ", objects_list)
    objects_list = objects_list.split(" ")
    objects_list = list(filter(None, objects_list))
    objects_list = list(set(objects_list))

    for i in range(len(objects_list)):
        objects_list[i] = singularize(objects_list[i]).lower()

    print("objects_list: ", objects_list)
    # covnver this list to a string separated by ","
    objects = ", ".join(objects_list)
    print("objects: ", objects)

    # boxes_filt, scores, pred_phrases = get_grounding_output(
    #     model, image, objects, box_threshold, text_threshold, device=device
    # )

    boxes_filt = None
    scores = None
    pred_phrases = None
    for i in range(len(objects_list)):
        word = objects_list[i]
        local_boxes_filt, local_scores, local_pred_phrases = get_grounding_output(
            model, image, word, box_threshold, text_threshold, device=device
        )
        print("pred_phrases: ", local_pred_phrases)
        if boxes_filt is None:
            boxes_filt = local_boxes_filt
            scores = local_scores
            pred_phrases = local_pred_phrases
        else:
            boxes_filt = torch.cat((boxes_filt, local_boxes_filt), 0)
            scores = torch.cat((scores, local_scores), 0)
            pred_phrases = pred_phrases + local_pred_phrases

    new_boxes_filt = convert_to_corners(boxes_filt)
    print(f"Before NMS: {boxes_filt.shape[0]} boxes")
    print(f"Before NMS: {pred_phrases} predictions")
    nms_idx = torchvision.ops.nms(new_boxes_filt, scores, iou_threshold).numpy().tolist()
    boxes_filt = boxes_filt[nms_idx]
    pred_phrases = [pred_phrases[idx] for idx in nms_idx]
    print(f"After NMS: {boxes_filt.shape[0]} boxes")
    print(f"After NMS: {pred_phrases} predictions")
    for i in range(boxes_filt.size(0)):
        print(f"Box {i}: {pred_phrases[i]} : {boxes_filt[i]}")


    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()

    # this is to get the caption about the image.
    # first collect all the objects.
    obj_dict = {}
    for i in range(len(pred_phrases)):
        obj = pred_phrases[i].split("(")[0]
        if obj in obj_dict:
            obj_dict[obj] = obj_dict[obj] + 1
        else:
            obj_dict[obj] = 1


    all_captions = {}
    all_captions["original"] = original_caption.text

    print("#" * 100)
    extra_detail = "This image has "
    for obj in obj_dict:
        extra_detail = extra_detail + str(obj_dict[obj]) + " " + obj + ", "

    extra_detail = extra_detail[:-2]
    extra_detail = extra_detail + " in it."
    print(extra_detail)


    # for the llm model
    if args.correct_with_llm:
        question = "The description of an image is given below. \n '" \
                   + original_caption.text + \
                   "' \n and now correct the description based on the following information of counts of objects in the image. \n" \
                   "" + extra_detail + \
                   "For example, if the description does not have any mention of say earphones, and the information of counts says there is one, add this information in the caption. \n " \
                   "If the description has gives information of 5 dogs, and the information of counts says there are 10 dogs, correct this information in the caption. \n " \
                   "If the description has gives information of 5 dogs, and the information of counts says there are 2 dogs, correct this information in the caption. \n and so on. "


        llm_corrected = text_model.generate_content(question, generation_config=genai.types.GenerationConfig(temperature=0.0))
        llm_corrected.resolve()

        all_captions["llm_corrected"] = llm_corrected.text


    # vlm re generation
    if args.correct_with_vlm:
        question = f"{caption_question} \n Note : {extra_detail}. Strictly follow the information given in the note. \n " \
                   f"For example if the note says there is an earphone mention it in the caption even if you do not see it. \n" \
                   f"If the note says there are 5 dogs, and if you see only 3 in them describe the 3 dogs and add add something like 'There are are 2 more dogs in the image' \n " \
                   f"and if the note says there are 5 dogs, and if you see 7 of them, then only describe the 5 dogs you are most sure of. \n " \
                   f"and so on. "

        vlm_corrected = vision_model.generate_content([question, image_pil])
        vlm_corrected.resolve()

        all_captions["vlm_corrected"] = vlm_corrected.text

    return all_captions, boxes_filt, pred_phrases


global text_model, vision_model, model, ram_model
global box_threshold, text_threshold, iou_threshold, device
global args

def save_captions(text_file, all_captions):
    with open(text_file, "w", encoding='utf-8') as f:
        f.write(f"Original Caption: \n")
        f.write("\n")
        f.write(all_captions["original"])
        f.write("\n")
        f.write("\n")
        f.write("\n")
        if args.correct_with_llm:
            f.write(f"Corrected with LLM: \n")
            f.write("\n")
            f.write(all_captions["llm_corrected"])
            f.write("\n")
            f.write("\n")
            f.write("\n")
        if args.correct_with_vlm:
            f.write(f"Corrected with VLM: \n")
            f.write("\n")
            f.write(all_captions["vlm_corrected"])
            f.write("\n")
            f.write("\n")
            f.write("\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--ram_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--split", default=",", type=str, help="split for text prompt")
    parser.add_argument("--openai_key", type=str, help="key for chatgpt")
    parser.add_argument("--openai_proxy", default=None, type=str, help="proxy for chatgpt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.25, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.2, help="text threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")

    parser.add_argument('--correct_with_llm', action='store_true', default=False)
    parser.add_argument('--correct_with_vlm', action='store_true', default=False)
    parser.add_argument('--original_inpaint', action='store_true', default=False)
    parser.add_argument('--dont_do_caption', action='store_true', default=False)

    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    ram_checkpoint = args.ram_checkpoint  # change the path of the model
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    image_path = args.input_image
    split = args.split
    openai_key = args.openai_key
    openai_proxy = args.openai_proxy
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    iou_threshold = args.iou_threshold
    device = args.device


    # make dir
    os.makedirs(output_dir, exist_ok=True)
    original_output_dir = f"{output_dir}/original"
    os.makedirs(original_output_dir, exist_ok=True)
    # load image

    image_pil, image = load_image(image_path)
    image_pil.save(os.path.join(original_output_dir, "raw_image.jpg"))
    image_path = os.path.join(original_output_dir, "raw_image.jpg")




    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)


    # initialize Recognize Anything Model
    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    transform = TS.Compose([
        TS.Resize((384, 384)),
        TS.ToTensor(), normalize
    ])
    ram_model = ram(pretrained=ram_checkpoint,
                    image_size=384,
                    vit='swin_l')
    ram_model.eval()
    ram_model = ram_model.to(device)


    GOOGLE_API_KEY = "AIzaSyCrR7Bd_4AVMizTjDwNqvDuBo0vOQLVGy8"
    genai.configure(api_key=GOOGLE_API_KEY)

    text_model = genai.GenerativeModel('gemini-pro')
    vision_model = genai.GenerativeModel('gemini-pro-vision')

    all_captions, boxes_filt, pred_phrases = get_caption_segmentation(image_pil, image, transform)

    # save the original caption and the new vision and text model captions to a file
    save_captions(f"{original_output_dir}/captions.txt", all_captions)





    # do this SAM thing once
    # initialize SAM
    if use_sam_hq:
        print("Initialize SAM-HQ Predictor")
        predictor = SamPredictor(build_sam_hq(checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    print(f"Predicting {transformed_boxes.shape} boxes")

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(device),
        multimask_output=False,
    )

    print(f"Predicted {masks.shape} masks")

    # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)


    plt.axis('off')
    plt.savefig(
        os.path.join(original_output_dir, "automatic_label_output.jpg"),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )
    plt.close()
    plt.clf()

    image_save = Image.fromarray(image)
    image_save.save(os.path.join(output_dir, "img.jpg"))


    blur_factor = 33
    seed = 0

    model_str = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe1 = AutoPipelineForInpainting.from_pretrained(model_str, torch_dtype=torch.float16)
    pipe1.enable_model_cpu_offload()

    model_str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    pipe2 = AutoPipelineForInpainting.from_pretrained(model_str, torch_dtype=torch.float16)
    pipe2.enable_model_cpu_offload()

    generator = torch.Generator(device='cuda').manual_seed(seed)

    expand_pixels = 10
    mask_threshold = 0.2

    original_caption = all_captions["original"]

    for i in range(len(masks)):

        mask = masks[i].cpu().numpy().astype(np.uint8) * 255
        mask = mask[0]  # Assuming mask shape is [1, height, width]

        # Convert to binary mask
        binary_mask = mask > 128  # Adjust the threshold as necessary

        labeled_array, num_features = scipy_label(binary_mask)
        new_mask = np.zeros_like(binary_mask)

        skip_mask = False
        for feature in range(1, num_features + 1):
            where = np.where(labeled_array == feature)
            min_y, max_y = np.min(where[0]), np.max(where[0])
            min_x, max_x = np.min(where[1]), np.max(where[1])

            width = max_x - min_x + 1
            height = max_y - min_y + 1

            if width <= 2 or height <= 2:
                continue

            min_y = max(min_y - expand_pixels, 0)
            max_y = min(max_y + expand_pixels, new_mask.shape[0] - 1)
            min_x = max(min_x - expand_pixels, 0)
            max_x = min(max_x + expand_pixels, new_mask.shape[1] - 1)

            new_mask[min_y:max_y + 1, min_x:max_x + 1] = 1

        if skip_mask:
            continue

        covered_area = np.sum(new_mask)
        total_image_area = np.sum(new_mask) + np.sum(1 - new_mask)

        print(f"Covered area: {covered_area}, Total image area: {total_image_area}")
        print(f"Mask threshold: {mask_threshold}")
        print(f"Covered area / Total image area: {covered_area / total_image_area}")

        if covered_area / total_image_area < mask_threshold and covered_area > 0.001:
            masked_directory = f"{output_dir}/masked_{i}"
            os.makedirs(masked_directory, exist_ok=True)

            mask_image = Image.fromarray(mask)
            mask_path = os.path.join(masked_directory, "mask.png")
            mask_image.save(mask_path)

            new_mask_image = Image.fromarray(new_mask.astype(np.uint8) * 255)
            new_mask_image.save(os.path.join(masked_directory, "block_mask.png"))
        else:
            print(f"Skipping masked_{i} as the covered area is more than 20% of the total image area.")
            continue

        # create dilated mask
        mask_uint8 = mask.astype(np.uint8) * 255
        blur_radius = 45
        blurred_mask = cv2.GaussianBlur(mask_uint8, (2 * blur_radius + 1, 2 * blur_radius + 1), 0)
        _, binary_mask_expanded = cv2.threshold(blurred_mask, 0, 255, cv2.THRESH_BINARY)
        binary_mask_expanded_bool = binary_mask_expanded.astype(bool)

        mask_expanded_image = Image.fromarray(binary_mask_expanded)
        mask_expanded_path = os.path.join(masked_directory, "mask_expanded.png")
        mask_expanded_image.save(mask_expanded_path)


        print(f"Processing masked_{i} and pred phrase {pred_phrases[i]}")
        object_to_remove = pred_phrases[i].split("(")[0]
        # negative_prompt = f"{object_to_remove}"
        # print(f"Negative prompt: {negative_prompt}")


        question = f"The description of the image is given by the following caption. \n"
        question += f"\"{original_caption}\" \n"
        question += f"Given the caption, we will now remove {object_to_remove} and replace it with another object. \n"
        question += "What objects would you like to replace it with? " \
                    f"Give less than 5 relevant suggestions to replace {object_to_remove} with separated by ','. \n" \
                    f"The objects should be of one word and should be different from {object_to_remove}, " \
                    f"but at the same time it should also be relevant to the image description after replacement. \n" \
                    f"If {object_to_remove} is a breed of an animal, then the replacement list should contain " \
                    f"2 suggestions of other breeds of the same animal. While remaining suggestions should be different" \
                    f"animals. \n" \

        print(question)

        response = text_model.generate_content(question,
                                               generation_config=genai.types.GenerationConfig(temperature=0.0))
        response.resolve()
        print(response.text)

        # objects_list = response.text.split(", ")
        objects_list = response.text
        objects_list = objects_list.replace(",", "")
        objects_list = objects_list.replace("\n", " ").replace("-", " ").replace(",", " ")

        pattern = "[^a-zA-Z0-9]"
        objects_list = re.sub(pattern, " ", objects_list)
        objects_list = objects_list.split(" ")
        objects_list = list(filter(None, objects_list))
        objects_list = list(set(objects_list))

        for i in range(len(objects_list)):
            objects_list[i] = singularize(objects_list[i]).lower()

        objects_list = list(set(objects_list))
        objects_list.append("")

        print(objects_list)

        negative_prompt = "Bad quality, bad composition, disfigured, mutated body parts"

        for i in range(len(objects_list)):
            print("#" * 50)
            object_to_replace_with = objects_list[i]

            if object_to_replace_with == "":
                prompt = f"A high quality picture, detailed, 8k uhd."
                print(f"Prompt: {prompt}")
            else:
                prompt = f"A high quality picture, detailed, 8k uhd. There is {object_to_replace_with} in the image."
                print(f"Prompt: {prompt}")



            strength = 0.99
            num_inference_steps = 200


            init_image = load_image_diffusers(os.path.join(output_dir, "img.jpg"))
            height = init_image.size[1]
            width = init_image.size[0]

            if args.original_inpaint:
                mask_image = load_image_diffusers(os.path.join(masked_directory, f"mask_expanded.png"))
                inpaint_image = pipe1(prompt=prompt, negative_prompt=negative_prompt, negative_prompt_2=negative_prompt,
                                         image=init_image, mask_image=mask_image,
                                         strength=strength, generator=generator, guidance_scale=7.5,
                                         num_inference_steps=num_inference_steps, height=height, width=width).images[0]
                print(inpaint_image)
                inpaint_image.save(os.path.join(masked_directory, f"inpaint_replace_{object_to_remove}_"
                                                                  f"with_{object_to_replace_with}_mask_expanded.jpg"))

                image_pil, image = load_image(os.path.join(masked_directory, f"inpaint_replace_{object_to_remove}_"
                                                                  f"with_{object_to_replace_with}_mask_expanded.jpg"))
                all_captions, _, _ = get_caption_segmentation(image_pil, image, transform)
                file = f"{masked_directory}/captions_replace_{object_to_remove}_with_{object_to_replace_with}_" \
                       f"mask_expanded.txt"
                save_captions(file, all_captions)




                mask_image = load_image_diffusers(os.path.join(masked_directory, f"mask_expanded.png"))
                mask_blurred = pipe1.mask_processor.blur(mask_image, blur_factor=blur_factor)
                mask_blurred.save(os.path.join(masked_directory, f"mask_expanded_blur.png"))

                inpaint_image = pipe1(prompt=prompt, negative_prompt=negative_prompt, negative_prompt_2=negative_prompt,
                                      image=init_image, mask_image=mask_blurred, padding_mask_crop=32,
                                      strength=strength, generator=generator, guidance_scale=7.5,
                                      num_inference_steps=num_inference_steps, height=height, width=width).images[0]
                print(inpaint_image)
                inpaint_image.save(os.path.join(masked_directory, f"inpaint_replace_{object_to_remove}_"
                                                                  f"with_{object_to_replace_with}_mask_expanded_blur.jpg"))

                image_pil, image = load_image(os.path.join(masked_directory, f"inpaint_replace_{object_to_remove}_"
                                                                  f"with_{object_to_replace_with}_mask_expanded_blur.jpg"))
                all_captions, _, _ = get_caption_segmentation(image_pil, image, transform)
                file = f"{masked_directory}/captions_replace_{object_to_remove}_with_{object_to_replace_with}_" \
                       f"mask_expanded_blur.txt"
                save_captions(file, all_captions)










            mask_image = load_image_diffusers(os.path.join(masked_directory, f"block_mask.png"))
            inpaint_image = pipe2(prompt=prompt, negative_prompt=negative_prompt, negative_prompt_2=negative_prompt,
                                  image=init_image, mask_image=mask_image,
                                  strength=strength, generator=generator, guidance_scale=7.5,
                                  num_inference_steps=num_inference_steps, height=height, width=width).images[0]
            print(inpaint_image)
            inpaint_image.save(os.path.join(masked_directory, f"inpaint_replace_{object_to_remove}_"
                                                              f"with_{object_to_replace_with}_block_mask.jpg"))

            image_pil, image = load_image(os.path.join(masked_directory, f"inpaint_replace_{object_to_remove}_"
                                                              f"with_{object_to_replace_with}_block_mask.jpg"))
            if args.dont_do_caption:
                pass
            else:
                all_captions, _, _ = get_caption_segmentation(image_pil, image, transform)
                file = f"{masked_directory}/captions_replace_{object_to_remove}_with_{object_to_replace_with}_" \
                       f"block_mask.txt"
                save_captions(file, all_captions)

            del inpaint_image
            del mask_image
            gc.collect()
            torch.cuda.empty_cache()



            mask_image = load_image_diffusers(os.path.join(masked_directory, f"block_mask.png"))
            inpaint_image = pipe2(prompt=prompt, negative_prompt=negative_prompt, negative_prompt_2=negative_prompt,
                                  image=init_image, mask_image=mask_image, padding_mask_crop=64,
                                  strength=strength, generator=generator, guidance_scale=7.5,
                                  num_inference_steps=num_inference_steps, height=height, width=width).images[0]
            print(inpaint_image)
            inpaint_image.save(os.path.join(masked_directory, f"inpaint_replace_{object_to_remove}_"
                                                              f"with_{object_to_replace_with}_block_mask_pipe_2_crop_64.jpg"))

            image_pil, image = load_image(os.path.join(masked_directory, f"inpaint_replace_{object_to_remove}_"
                                                                         f"with_{object_to_replace_with}_block_mask"
                                                                         f"_pipe_2_crop_64.jpg"))
            if args.dont_do_caption:
                pass
            else:
                all_captions, _, _ = get_caption_segmentation(image_pil, image, transform)
                file = f"{masked_directory}/captions_replace_{object_to_remove}_with_{object_to_replace_with}_" \
                       f"block_mask_pipe_2_crop_64.txt"
                save_captions(file, all_captions)

            del inpaint_image
            del mask_image
            gc.collect()
            torch.cuda.empty_cache()



            mask_image = load_image_diffusers(os.path.join(masked_directory, f"block_mask.png"))
            inpaint_image = pipe1(prompt=prompt, negative_prompt=negative_prompt, negative_prompt_2=negative_prompt,
                                  image=init_image, mask_image=mask_image,
                                  strength=strength, generator=generator, guidance_scale=7.5,
                                  num_inference_steps=num_inference_steps, height=height, width=width).images[0]
            print(inpaint_image)
            inpaint_image.save(os.path.join(masked_directory, f"inpaint_replace_{object_to_remove}_"
                                                              f"with_{object_to_replace_with}_block_mask_1.jpg"))

            image_pil, image = load_image(os.path.join(masked_directory, f"inpaint_replace_{object_to_remove}_"
                                                                         f"with_{object_to_replace_with}_block_mask_1.jpg"))
            if args.dont_do_caption:
                pass
            else:
                all_captions, _, _ = get_caption_segmentation(image_pil, image, transform)
                file = f"{masked_directory}/captions_replace_{object_to_remove}_with_{object_to_replace_with}_" \
                       f"block_mask_1.txt"
                save_captions(file, all_captions)

            del inpaint_image
            del mask_image
            gc.collect()
            torch.cuda.empty_cache()


            padding_mask_crop = 64
            mask_image = load_image_diffusers(os.path.join(masked_directory, f"block_mask.png"))
            inpaint_image = pipe1(prompt=prompt, negative_prompt=negative_prompt, negative_prompt_2=negative_prompt,
                                  image=init_image, mask_image=mask_image, padding_mask_crop=padding_mask_crop,
                                  strength=strength, generator=generator, guidance_scale=7.5,
                                  num_inference_steps=num_inference_steps, height=height, width=width).images[0]
            print(inpaint_image)
            inpaint_image.save(os.path.join(masked_directory, f"inpaint_replace_{object_to_remove}_"
                                                              f"with_{object_to_replace_with}_block_mask_crop_"
                                                              f"{padding_mask_crop}.jpg"))

            image_pil, image = load_image(os.path.join(masked_directory, f"inpaint_replace_{object_to_remove}_"
                                                                         f"with_{object_to_replace_with}_block_mask_"
                                                                         f"crop_{padding_mask_crop}.jpg"))
            if args.dont_do_caption:
                pass
            else:
                all_captions, _, _ = get_caption_segmentation(image_pil, image, transform)
                file = f"{masked_directory}/captions_replace_{object_to_remove}_with_{object_to_replace_with}_" \
                       f"block_mask_crop_{padding_mask_crop}.txt"
                save_captions(file, all_captions)

            del inpaint_image
            del mask_image
            gc.collect()
            torch.cuda.empty_cache()




            local_strength = 0.70
            lss = "0_70"
            local_steps = int(num_inference_steps / local_strength)

            # reinpaint the previous inpaint image with less strength
            mask_image = load_image_diffusers(os.path.join(masked_directory, f"block_mask.png"))
            mask_image = pipe1.mask_processor.blur(mask_image, blur_factor=5)
            init_image_inpaint = load_image_diffusers(os.path.join(masked_directory, f"inpaint_replace_"
                                                                                     f"{object_to_remove}_with_"
                                                                                     f"{object_to_replace_with}_"
                                                                                     f"block_mask_crop_"
                                                                                     f"{padding_mask_crop}.jpg"))


            inpaint_image = pipe2(prompt=prompt, negative_prompt=negative_prompt, negative_prompt_2=negative_prompt,
                                  image=init_image_inpaint, mask_image=mask_image,
                                  strength=local_strength, generator=generator, guidance_scale=7.5,
                                  num_inference_steps=local_steps, height=height, width=width).images[0]
            print(inpaint_image)
            inpaint_image.save(os.path.join(masked_directory, f"inpaint_replace_{object_to_remove}_"
                                                              f"with_{object_to_replace_with}_block_mask_crop_"
                                                              f"{padding_mask_crop}_"
                                                              f"reinpaint_{lss}.jpg"))

            image_pil, image = load_image(os.path.join(masked_directory, f"inpaint_replace_{object_to_remove}_"
                                                                         f"with_{object_to_replace_with}_block_mask_crop"
                                                                         f"_{padding_mask_crop}_reinpaint_{lss}.jpg"))
            if args.dont_do_caption:
                pass
            else:
                all_captions, _, _ = get_caption_segmentation(image_pil, image, transform)
                file = f"{masked_directory}/captions_replace_{object_to_remove}_with_{object_to_replace_with}_" \
                       f"block_mask_crop_{padding_mask_crop}_reinpaint_{lss}.txt"
                save_captions(file, all_captions)

            del inpaint_image
            del mask_image
            gc.collect()
            torch.cuda.empty_cache()



            # reinpaint twice the previous inpaint image with less strength
            mask_image = load_image_diffusers(os.path.join(masked_directory, f"block_mask.png"))
            mask_image = pipe1.mask_processor.blur(mask_image, blur_factor=15)
            init_image_inpaint = load_image_diffusers(os.path.join(masked_directory, f"inpaint_replace_"
                                                                                     f"{object_to_remove}_with_"
                                                                                     f"{object_to_replace_with}_"
                                                                                     f"block_mask_crop_"
                                                                                     f"{padding_mask_crop}"
                                                                                     f"_reinpaint_{lss}.jpg"))
            inpaint_image = pipe2(prompt=prompt, negative_prompt=negative_prompt, negative_prompt_2=negative_prompt,
                                  image=init_image_inpaint, mask_image=mask_image,
                                  strength=local_strength, generator=generator, guidance_scale=7.5,
                                  num_inference_steps=local_steps, height=height, width=width).images[0]
            print(inpaint_image)
            inpaint_image.save(os.path.join(masked_directory, f"inpaint_replace_{object_to_remove}_"
                                                              f"with_{object_to_replace_with}_block_mask_crop_"
                                                              f"{padding_mask_crop}_"
                                                              f"reinpaint_{lss}_reinpaint_{lss}.jpg"))

            image_pil, image = load_image(os.path.join(masked_directory, f"inpaint_replace_{object_to_remove}_"
                                                                         f"with_{object_to_replace_with}_block_mask_crop"
                                                                         f"_{padding_mask_crop}"
                                                                         f"_reinpaint_{lss}_reinpaint_{lss}.jpg"))
            if args.dont_do_caption:
                pass
            else:
                all_captions, _, _ = get_caption_segmentation(image_pil, image, transform)
                file = f"{masked_directory}/captions_replace_{object_to_remove}_with_{object_to_replace_with}_" \
                       f"block_mask_crop_{padding_mask_crop}_reinpaint_{lss}_reinpaint_{lss}.txt"
                save_captions(file, all_captions)


            del inpaint_image
            del mask_image
            gc.collect()
            torch.cuda.empty_cache()



            # # reinpaint with blur mask now
            # mask_image = load_image_diffusers(os.path.join(masked_directory, f"block_mask.png"))
            # mask_blurred = pipe1.mask_processor.blur(mask_image, blur_factor=blur_factor)
            # mask_blurred.save(os.path.join(masked_directory, f"block_mask_blur.png"))
            #
            # init_image_inpaint = load_image_diffusers(os.path.join(masked_directory, f"inpaint_replace_"
            #                                                                          f"{object_to_remove}_with_"
            #                                                                          f"{object_to_replace_with}_"
            #                                                                          f"block_mask_crop_32"
            #                                                                          f"_reinpaint_{lss}"
            #                                                                          f"_reinpaint_{lss}.jpg"))
            # inpaint_image = pipe1(prompt=prompt, negative_prompt=negative_prompt, negative_prompt_2=negative_prompt,
            #                       image=init_image_inpaint, mask_image=mask_blurred,
            #                       strength=local_strength, generator=generator, guidance_scale=7.5,
            #                       num_inference_steps=local_steps, height=height, width=width).images[0]
            # print(inpaint_image)
            # inpaint_image.save(os.path.join(masked_directory, f"inpaint_replace_{object_to_remove}_"
            #                                                   f"with_{object_to_replace_with}_block_mask_crop_32_"
            #                                                   f"reinpaint_{lss}_reinpaint_{lss}_reinpaint_blur_{lss}.jpg"))
            #
            # image_pil, image = load_image(os.path.join(masked_directory, f"inpaint_replace_{object_to_remove}_"
            #                                                              f"with_{object_to_replace_with}_block_mask_crop"
            #                                                              f"_32_reinpaint_{lss}_reinpaint_{lss}"
            #                                                              f"_reinpaint_blur_{lss}.jpg"))
            # all_captions, _, _ = get_caption_segmentation(image_pil, image, transform)
            # file = f"{masked_directory}/captions_replace_{object_to_remove}_with_{object_to_replace_with}_" \
            #        f"block_mask_crop_32_reinpaint_{lss}_reinpaint_{lss}_reinpaint_blur_{lss}.txt"
            # save_captions(file, all_captions)
            #
            # del inpaint_image
            # del mask_image
            # gc.collect()
            # torch.cuda.empty_cache()




            # mask_image = load_image_diffusers(os.path.join(masked_directory, f"block_mask.png"))
            # inpaint_image = pipe1(prompt=prompt, negative_prompt=negative_prompt, negative_prompt_2=negative_prompt,
            #                       image=init_image, mask_image=mask_image, padding_mask_crop=9,
            #                       strength=strength, generator=generator, guidance_scale=7.5,
            #                       num_inference_steps=num_inference_steps, height=height, width=width).images[0]
            # print(inpaint_image)
            # inpaint_image.save(os.path.join(masked_directory, f"inpaint_replace_{object_to_remove}_"
            #                                                   f"with_{object_to_replace_with}_block_mask_crop_9.jpg"))
            #
            # image_pil, image = load_image(os.path.join(masked_directory, f"inpaint_replace_{object_to_remove}_"
            #                                                              f"with_{object_to_replace_with}_block_mask_crop_9.jpg"))
            # all_captions, _, _ = get_caption_segmentation(image_pil, image, transform)
            # file = f"{masked_directory}/captions_replace_{object_to_remove}_with_{object_to_replace_with}_" \
            #        f"block_mask_crop_9.txt"
            # save_captions(file, all_captions)
            #
            # del inpaint_image
            # del mask_image
            # gc.collect()
            # torch.cuda.empty_cache()



            # mask_image = load_image_diffusers(os.path.join(masked_directory, f"block_mask.png"))
            # mask_blurred = pipe1.mask_processor.blur(mask_image, blur_factor=blur_factor)
            # mask_blurred.save(os.path.join(masked_directory, f"block_mask_blur.png"))
            #
            # inpaint_image = pipe1(prompt=prompt, negative_prompt=negative_prompt, negative_prompt_2=negative_prompt,
            #                       image=init_image, mask_image=mask_blurred, padding_mask_crop=32,
            #                       strength=strength, generator=generator, guidance_scale=7.5,
            #                       num_inference_steps=num_inference_steps, height=height, width=width).images[0]
            # print(inpaint_image)
            # inpaint_image.save(os.path.join(masked_directory, f"inpaint_replace_{object_to_remove}_"
            #                                                   f"with_{object_to_replace_with}_block_mask_blur_32.jpg"))
            #
            # image_pil, image = load_image(os.path.join(masked_directory, f"inpaint_replace_{object_to_remove}_"
            #                                                   f"with_{object_to_replace_with}_block_mask_blur_32.jpg"))
            # all_captions, _, _ = get_caption_segmentation(image_pil, image, transform)
            # file = f"{masked_directory}/captions_replace_{object_to_remove}_with_{object_to_replace_with}_" \
            #        f"block_mask_blur_32.txt"
            # save_captions(file, all_captions)
            #
            # del inpaint_image
            # del mask_image
            # gc.collect()
            # torch.cuda.empty_cache()



















