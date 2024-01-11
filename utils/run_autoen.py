from models import Autoencoders
import torchvision
import preprocessor
import glob
from tqdm import tqdm

model=Autoencoders.Simple_autoencoder()
all_spec=glob.glob("/Users/kunkerdthaisong/ipu/ntu_rgb_proj/spec/*.png",recursive=True)
for i in tqdm(all_spec):
    ten_img=preprocessor.preprocess(i)
    output_tensor = model(ten_img.unsqueeze(0))
    output_image = torchvision.transforms.ToPILImage()(output_tensor.squeeze(0))
    output_image=torchvision.transforms.Resize(334)(output_image) #origin size  #type PIL.Image.Image
    output_image.save(i)