




# from ultralytics import YOLO
# if __name__ == "__main__":
#     model = YOLO("ultralytics/models/v8/model.yaml")
#     model.train(**{'cfg':'ultralytics/yolo/cfg/default.yaml'})


from ultralytics import YOLO
if __name__ == "__main__":

# "From pretrained(recommended)"
# 加载了预训练模型，然后再进行5轮训练。
#     model = YOLO('yolov8n.pt') # pass any model type
    # model.train(epochs=5)


# "From scratch"
    #从0开始训练，没有预训练模型
	#根据yolov8n.yaml的这文件重新搭建一个新的模型。
    model = YOLO('ultralytics/models/v8/yolov8-fasternet.yaml')
    # #然后从头开始训练。
    model.train(**{'cfg':'ultralytics/yolo/cfg/default.yaml'})


# "Resume"
    #因为一些原因（停电等等）中断了之前的训练，然后想继续之前的训练，用resume。
    # from ultralytics import YOLO
    # model = YOLO("last.pt")
    # model.train(resume=True)


