import pathlib


def print_model(model_file_path, model_weights, coefficient, predictor):
    weights_to_write = "[COEFFICIENT] * " + str(coefficient) + " + \n"

    for feature, weight in zip(model_weights.keys(), model_weights.values()):
        weights_to_write += "[" + feature + "] * " + str(weight) + " + \n"

    text_file_path = model_file_path + str(predictor) + 'model1.txt'
    text_file = open(text_file_path, 'w')
    text_file.write(weights_to_write)
    text_file.close()
