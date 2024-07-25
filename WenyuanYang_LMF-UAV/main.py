







# from ultralytics import YOLOv10
#
# model = YOLOv10()
# # If you want to finetune the model with pretrained weights, you could load the
# # pretrained weights like below
# # model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')
# # or
# # wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
# # model = YOLOv10('yolov10{n/s/m/b/l/x}.pt')
#
# model.train(data='VisDrone.yaml', epochs=200, batch=1, imgsz=640)



from ultralytics import YOLOv10

if __name__ == "__main__":
    model = YOLOv10("ultralytics/cfg/models/v10/111n.yaml")
    model.train(**{'cfg':'ultralytics/cfg/default.yaml', 'data':'datasets/VisDrone.yaml'})