from Swap import resize_image,\
    segment_preprocesses,\
    change_skin_color,\
    remove_head_and_hair,\
    translate_face,\
    apply_inpainting,\
    change_solid_bw,\
    apply_seamless_cloning,\
    find_lip_loc
#from swap.utils import change_solid_bw

from Segmentation import segment_image
import cv2
import numpy as np

# TODO: Remove this function
def get_full_mhp(image_name):
    image_path = "./images/"
    input_image = cv2.imread(image_path + image_name)
    full = segment_image(input_image)
    return full

def main(model_image, user_image, schp_model_path):
    # Remove solid colors (If needed)
    model_image = change_solid_bw(model_image)
    user_image = change_solid_bw(user_image)
    
    model_image, lip_user = resize_image(model_image, user_image)
  
    model_image, user_image, model_segment, user_segment, w = segment_preprocesses(model_image, user_image, schp_model_path)


    
    lip_loc = find_lip_loc(lip_user, w)
  
    model_image = change_skin_color(model_image, user_image, model_segment, user_segment)

    remove_head_and_hair(model_image, model_segment)
    
    user_image, user_segment, translated_lip = translate_face(model_segment, user_segment, model_image, user_image, lip_loc)
   
    # sem_res = apply_seamless_cloning(model_image, user_image, user_segment)
    
    inpaint_image, mask = apply_inpainting(model_image, translated_lip)

    # return sem_res, user_image, model_image, user_segment, model_segment
    return inpaint_image, mask, user_image, model_image, user_segment, model_segment


# if __name__ == "__main__":
#     image_path = "./images/"
#
#     args = argparse.ArgumentParser()
#     args.add_argument("--model_image", type=str, default="model.jpg")
#     args.add_argument("--user_image", type=str, default="model.jpg")
#
#     # "/content/drive/MyDrive/SCHP/exp_schp_multi_cihp_local.pth"
#     args.add_argument("--schp_model_path", type=str, required=True)
#
#     opt = args.parse_known_args()[0]
#
#     model_image = cv2.imread(image_path + opt.model_image)
#     user_image = cv2.imread(image_path + opt.user_image)
#
#     main(model_image, user_image, opt.schp_model_path)
