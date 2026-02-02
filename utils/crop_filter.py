def filter_predictions_by_crop(predictions, selected_crop):
    """
    predictions = [(label, prob), ...]
    selected_crop = "Rice" / "Maize" / "Cotton" / "Wheat" / "Sugarcane"
    """

    crop_key = selected_crop.lower()

    filtered = [
        (label, prob)
        for label, prob in predictions
        if crop_key in label.lower()
    ]

    if len(filtered) == 0:
        return predictions  # fallback

    return sorted(filtered, key=lambda x: x[1], reverse=True)
