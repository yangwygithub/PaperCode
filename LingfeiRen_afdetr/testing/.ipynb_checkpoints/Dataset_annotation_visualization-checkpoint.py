# 标注可视化
from pycocotools.coco import COCO
import numpy as np
import os.path as osp
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from PIL import Image


def apply_exif_orientation(image):
    _EXIF_ORIENT = 274
    if not hasattr(image, 'getexif'):
        return image
    
    try:
        exif = image.getexif()
    except Exception:
        exif = None

    if exif is None:
        return image
    
    orientation = exif.get(_EXIF_ORIENT)

    method = {
        2: Image.FLIP_LEFT_RIGHT,
        3: Image.ROTATE_180,
        4: Image.FLIP_TOP_BOTTOM,
        5: Image.TRANSPOSE,
        6: Image.ROTATE_270,
        7: Image.TRANSVERSE,
        8: Image.ROTATE_90,
    }.get(orientation)

    if method is not None:
        return image.transpose(method)
    return image


def show_bbox_only(coco, anns, show_label_bbox=True, is_filling=True):
    "show bounding box of annotations only."
    if len(anns) == 0:
        return
    
    ax = plt.gca()
    ax.set_autoscale_on(False)

    image2color = dict()
    for cat in coco.getCatIds():
        # image2color[cat] = (np.random.random(1,3) * 0.7 +0.3).tolist()[0]
        image2color[cat] = (np.random.random(3,) * 0.7 +0.3).tolist()

    polygons = []
    colors = []

    for ann in anns:
        color = image2color[ann['category_id']]
        bbox_x, bbox_y, bbox_w, bbox_h = ann['bbox']
        poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y]]
        polygons.append(Polygon(np.array(poly).reshape((4,2))))
        colors.append(color)

        if show_label_bbox:
            label_bbox = dict(facecolor=color)
        else:
            label_bbox = None

        ax.text(
            bbox_x,
            bbox_y,
            '%s' % (coco.loadCats(ann['category_id'])[0]['name']),
            color = 'white',
            bbox=label_bbox
        )
    
    if is_filling:
        p = PatchCollection(
            polygons, facecolor=colors, linewidths=0, alpha=0.4
        )
        ax.add_collection(p)

coco = COCO('/root/autodl-tmp/code/VOC_dateset/VisDroneVehicle/annotations/train.json')
image_ids = coco.getImgIds()
# print(image_ids)
np.random.shuffle(image_ids)

# coco_img_ids = set(coco.imgs.keys())
# invalid_ids = set(image_ids) - coco_img_ids
# print('invalid_ids=', invalid_ids)

# for id in image_ids:
#     img_info = coco.loadImgs(id)[0]
#     img_path = os.path.join(coco.root, img_info['file_name'])
#     if not os.path.exists(img_path):
#         print(f"Image {img_path} missing!")

plt.figure(figsize=(16,5))
# 可视化8张图片
for i in range(8):
    print(image_ids[i])
    image_data = coco.loadImgs(image_ids[i])[0]
    image_path = osp.join('/root/autodl-tmp/code/VOC_dateset/VisDroneVehicle/train/images/', image_data['file_name'])
    filename = image_data['file_name']
    annotation_ids = coco.getAnnIds(
        imgIds=image_data['id'], catIds=[], iscrowd=0
    )
    annotations = coco.loadAnns(annotation_ids)

    ax = plt.subplot(2, 4, i+1)
    image = Image.open(image_path).convert("RGB")

    # important
    image=apply_exif_orientation(image)

    ax.imshow(image)
    # print(annotations.__dict__)
    print("annotations",annotations)
    show_bbox_only(coco, annotations)

    plt.title(f'{filename}')
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.savefig('annotations.png')