from ultralytics import YOLO

if __name__ == '__main__':
    models = [
        ('fire',r'C:\Users\39608\Desktop\demo1\data\fire1\fire.yaml'),
        # ('face',r'C:\Users\39608\Desktop\demo1\data\face\face.yaml')
    ]
    weight = r'yolov8n'
    for name, data_yaml in models:

        model = YOLO(weight)
        model.train(
            data=data_yaml,
            epochs=100,
            workers=2,
            batch=2,
            optimizer='SGD',
            lr0=0.000001,
            lrf=0.01,
            imgsz=640,
            freeze=10,
            augment=True,
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.3,  
            patience=15, 
            device = '0',
            amp = True,
            name=f"{name}训练成功"

        )