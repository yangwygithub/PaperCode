











from ultralytics import YOLOv10

if __name__ == "__main__":
    model = YOLOv10("ultralytics/cfg/models/v10/yolov10b.yaml")
    model.train(**{'cfg':'ultralytics/cfg/default.yaml', 'data':'datasets/VisDrone.yaml'})


    # model = YOLOv10("best.pt")
    # model.export(format='onnx', opset=13)


    # model = YOLOv10("best.pt")
    # model.predict(source='D:/VisDrone/VisDrone2019-DET-test-dev/images', **{'save': True})
    # model.predict(**{'cfg':'ultralytics/cfg/default.yaml', 'source':'111.jpg'})







