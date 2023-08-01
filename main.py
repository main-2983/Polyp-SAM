import numpy as np
from PIL import Image

from segment_anything import build_sam_vit_b, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide


image_path = "/mnt/sdd/nguyen.van.quan/Researchs/Polyp/Dataset/TestDataset/ETIS-LaribPolypDB/images/1.png"
image = Image.open(image_path)
image = np.asarray(image)

sam = build_sam_vit_b(checkpoint="ckpts/sam_vit_b_01ec64.pth")
sam = sam.to("cuda")

predictor = SamPredictor(sam)

predictor.set_image(image)

print("A")