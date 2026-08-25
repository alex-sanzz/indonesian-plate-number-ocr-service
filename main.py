import simplelpr
from fastapi import FastAPI, UploadFile, File
import shutil
app = FastAPI()



@app.post("/api/plate-number-ocr")
async def plateNumberOcr(file: UploadFile = File(...)):
    with open("./image.jpg", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    result = detectPlateNumber()
    return {
        "plate-text" : result
    }

def detectPlateNumber():
    # Initialize the engine
    setup_params = simplelpr.EngineSetupParms()
    engine = simplelpr.SimpleLPR(setup_params)

    # Configure for your country (e.g., UK = 90)
    engine.set_countryWeight(42, 1.0)
    engine.realizeCountryWeights()

    # Create a processor
    processor = engine.createProcessor()

    # Analyze an image
    candidates = processor.analyze("image.jpg")

    highest_confidence = 0
    match_plate = ""
    # Print results
    for candidate in candidates:
        for match in candidate.matches:
            if match.confidence > highest_confidence:
                highest_confidence = match.confidence
                match_plate = match.text
    return match_plate





# model_path = hf_hub_download(
#     repo_id="morsetechlab/yolov11-license-plate-detection",
#     filename="license-plate-finetune-v1x.pt"
# )

# model = YOLO(model_path)

# imgname = "trial4.jpg"

# results = model(imgname)

# # for result in results -> results contains all of images which are examined
# # result.boxes -> contains all of bounding boxes that mark desired object
# best_box = max(results[0].boxes, key=lambda box: (
    
#     float(box.xyxy[0][2] - box.xyxy[0][0]) * float(box.xyxy[0][3] - box.xyxy[0][1])
# ))

# x1, y1, x2, y2 = map(int, best_box.xyxy[0])

# image = cv2.imread(imgname)

# plate = image[y1:y2, x1:x2]

# plate = cv2.resize(
#     plate, 
#     None,
#     fx=3,
#     fy=3,
#     interpolation=cv2.INTER_CUBIC
# )

# cv2.imwrite("debug_plate.jpg", plate)

# ocr = PaddleOCR(
#     lang="en"
# )

# result = ocr.predict(plate)

# print(result)

        

        