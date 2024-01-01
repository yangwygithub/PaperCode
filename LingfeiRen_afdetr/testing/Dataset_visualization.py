# 数据集可视化
import os
import matplotlib.pyplot as plt
from PIL import Image

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

oringinal_images = []
images = []
texts = []
plt.figure(figsize=(16,5))

image_paths = [filename for filename in os.listdir('/root/autodl-tmp/code/VOC_dateset/VisDrone/train_int')][:8] # 取8张图片做可视化

for i,filename in enumerate(image_paths):
    name = os.path.splitext(filename)[0]

    image = Image.open('/root/autodl-tmp/code/VOC_dateset/VisDrone/train_int/'+filename).convert("RGB")

    plt.subplot(2,4,i+1)
    plt.imshow(image)
    plt.title(f'{filename}')
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()

plt.savefig('dataset_vis.png')

plt.show
