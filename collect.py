import cv2
import os

dir = "./outputs/demo9/"
raw_image = cv2.imread(dir + "img.jpg")
labels = cv2.imread(dir + "/original/automatic_label_output.jpg")

# resise the labels to the same size as the raw image
labels = cv2.resize(labels, (raw_image.shape[1], raw_image.shape[0]))

couple = cv2.hconcat([raw_image, labels])

# get all folders in the dir that start with "masked"
folders = [f for f in os.listdir(dir) if f.startswith("masked")]
for folder in folders:
    mask = cv2.imread(dir + folder + "/mask.png")
    mask = cv2.resize(mask, (raw_image.shape[1], raw_image.shape[0]))
    mask_inpaint = cv2.imread(dir + folder + "/inpaint_mask_image.jpg")
    mask_inpaint = cv2.resize(mask_inpaint, (raw_image.shape[1], raw_image.shape[0]))

    save_img = cv2.hconcat([couple, mask, mask_inpaint])
    cv2.imwrite(dir + folder + "/mask_inpaint_all.jpg", save_img)

    mask = cv2.imread(dir + folder + "/new_mask.png")
    mask = cv2.resize(mask, (raw_image.shape[1], raw_image.shape[0]))
    mask_inpaint = cv2.imread(dir + folder + "/inpaint_bold_mask_image_with_new_caption_refined.jpg")
    mask_inpaint = cv2.resize(mask_inpaint, (raw_image.shape[1], raw_image.shape[0]))

    save_img = cv2.hconcat([couple, mask, mask_inpaint])
    cv2.imwrite(dir + folder + "/mask_boxed_all.jpg", save_img)
