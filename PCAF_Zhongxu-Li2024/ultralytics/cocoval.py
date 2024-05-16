
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

if __name__ == '__main__':
	anno_json = 'D:/coco/val2017/annotations/instances_val2017.json'
	# anno_json = 'D:/code/yolov8/ultralytics-main/datasets/coco/annotations/instances_val2017.json'
	pred_json = 'D:/threeenglish/v8/ultralytics/runs/detect/threepapermain/train15/predictions.json'

	# 使用COCO API加载预测结果和标注
	cocoGt = COCO(anno_json)
	cocoDt = cocoGt.loadRes(pred_json)

	# 创建COCOeval对象
	cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')

	# 执行评估
	cocoEval.evaluate()
	cocoEval.accumulate()
	cocoEval.summarize()

	# 保存结果
	with open('./output/coco_eval.txt', 'w') as f:
		f.write(str(cocoEval.stats))

	# 打印结果
	print(cocoEval.stats)