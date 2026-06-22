from ultralytics import YOLO


"""
show landmarks od trained YOLO model in a image and validate the model

Input:
    - model_path: path to trained YOLO model
    - path to image(s) that should be predicted
"""
def seeModelInImg(model_path, img_path):

    #load model
    if isinstance(model_path, YOLO):
        model = model_path
    else:
        model = YOLO(model_path)
    #give image(s) into the model
    results = model(img_path)

    #show image(s)
    for result in results:
        result.show()

"""
compute results of YOLO model

Input:
- model_path: path to trained YOLO model
- img: image with RGB values
"""
def predictYOLO(model, img):
    results = model(img)

    return results # For more information about YOLO result output visit

