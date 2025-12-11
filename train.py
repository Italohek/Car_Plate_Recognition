from ultralytics import YOLO # type: ignore
import torch

def main():
    model = YOLO("yolov8n.pt")

    """ 
    Treina o modelo
    data = caminho pro data.yaml
    epochs = número de épocas total
    imgsz = tamanho da imagem (640 é o padrão aqui)
    device = dispositivo utilizdo. Aqui no caso vamos utilizar uma rtx4060 
    """

    results = model.train(
        data="dataset/data.yaml", 
        epochs=20, 
        imgsz=640, 
        device="0",

        degrees=10.0,   
        translate=0.1,
        scale=0.5, 
        shear=0.1,
        )

if __name__ == "__main__":
    main()