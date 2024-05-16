


import warnings
warnings.filterwarnings('ignore')


from ultralytics import YOLO

if __name__ == "__main__":
    # model = YOLO("ultralytics/cfg/models/v8/yolov8gaijin2.yaml")
    # model.train(**{'cfg':'ultralytics/cfg/default.yaml', 'data':'datasets/VisDrone.yaml'})



 # 模型验证
 #    model = YOLO("runs/detect/threepapermain/train7/weights/best.pt")
    # model.val(**{'data':'datasets/VisDrone.yaml'})

#  模型推理
    model = YOLO("runs/detect/threepapermain/train11/weights/best.pt")
    model.predict(source='D:/888/v8/ultralytics/yolo/v8/detect/dronevehicler', **{'save':True})

    #
    # model = YOLO("runs/detect/threepapermain/train3/weights/best.pt")
    # model.export(format='onnx', opset=14)



# from ultralytics import YOLO
# if __name__ == "__main__":
#
# "From pretrained(recommended)"
# 加载了预训练模型，然后再进行5轮训练。
#     model = YOLO('yolov8n.pt') # pass any model type
#     model.train(epochs=5)

#
# # "From scratch"
#     #从0开始训练，没有预训练模型
# 	#根据yolov8n.yaml的这文件重新搭建一个新的模型。
#     model = YOLO('ultralytics/models/v8/1_n.yaml')
#     # #然后从头开始训练。
#     model.train(**{'cfg':'ultralytics/yolo/cfg/default.yaml'})


# "Resume"
    #因为一些原因（停电等等）中断了之前的训练，然后想继续之前的训练，用resume。
    # from ultralytics import YOLO
    # model = YOLO("last.pt")
    # model.train(resume=True)



# from ultralytics import YOLO
#
# # Load a model
# # model = YOLO('yolov8n.pt')  # load an official model
# model = YOLO('runs/detect/train15/weights/best.pt')  # load a custom trained
#
# # Export the model
# model.export(format='onnx')