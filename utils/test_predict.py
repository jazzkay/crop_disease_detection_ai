from predict import predict_image

label, conf = predict_image("test.jpg")
print("Prediction:", label)
print("Confidence:", conf)
