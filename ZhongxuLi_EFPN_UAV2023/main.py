# from ultralytics import YOLO


from ultralytics import YOLO
if __name__ == "__main__":
    model = YOLO("models/e-fpn-n.yaml")
    model.train(**{'cfg':'ultralytics/yolo/cfg/default.yaml'})




