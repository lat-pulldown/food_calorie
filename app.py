import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions

# Load the pre-trained MobileNet model
model = MobileNet(weights='imagenet')

def predict_image(img_file):
    """
    Predict the top classe for an uploaded image using MobileNet.
    
    Args:
        img_file: File-like object representing the uploaded image.
    
    Returns:
        List of dictionaries containing 'label' and 'score' for the top prediction.
    """
    # Load image from file-like object
    img = image.load_img(img_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make predictions
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    
    # Extract labels and scores
    result = [
        {'label': label, 'score': round(score, 2)}
        for _, label, score in decoded_predictions
    ]
    return result