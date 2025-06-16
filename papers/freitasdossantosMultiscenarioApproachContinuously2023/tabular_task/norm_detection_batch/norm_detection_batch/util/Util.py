import math
import numpy as np
from sklearn.cluster import KMeans

from norm_detection_batch.lmt import PredictionResults, LMTSettings
from norm_detection_batch.lmt.ModelWeights import ModelWeights
from norm_detection_batch.lmt.RankAttribute import RankAttribute
from norm_detection_batch.features_aux.Features import Features
from norm_detection_batch.features_aux.FeatureImportance import FeatureImportance
from experiments.ensemble_training_66_perc import EnsembleTraining66Perc
from norm_detection_batch.util import UtilExcel


def get_model_weights(model_file_path, model_classify_regular_edition: False):
    """
    Method responsible for getting the weights for the models. In in this case, the values of the logistic regression equation.
    :param model_file_path: the path of the file that contains the weights for the models
    :param model_classify_regular_edition: True if we want to change our model to calculate the probability of
    classifying the edition as a Regular Edition (instead of Vandalism). This is important because we also have the
    case in which the feedback from the community changes an edition that was previously classified as Regular to
    Vandalism.
    :return: Return the weights for the models, these weights are used then to calculate the estimate for
        vandalism or regular edition.
    """

    my_model_weights = ModelWeights()

    model_file = open(model_file_path, 'r')
    lines = model_file.readlines()

    for line_aux in lines:
        index_last_marker = line_aux.find(']', 1)
        feature_name = line_aux[1: index_last_marker]

        index_marker_multiple = line_aux.find('*', index_last_marker)
        index_marker_plus = line_aux.find('+', index_marker_multiple)
        feature_weight = line_aux[index_marker_multiple + 1: index_marker_plus].strip()
        if model_classify_regular_edition:
            # Multiply by -1 because here we want the probability of classifying as regular.
            feature_weight = float(feature_weight) * (-1)

        setattr(my_model_weights, feature_name, float(feature_weight))

    model_file.close()
    return my_model_weights


def get_edition_features(edition_file_path):
    my_edition_features = Features()

    edition_file = open(edition_file_path, 'r')
    lines = edition_file.readlines()

    all_features_names = lines[0].split()
    all_features_values = lines[1].split()

    for (feature_name, feature_value) in zip(all_features_names, all_features_values):
        feature_value = get_casted_feature_value(feature_value)
        setattr(my_edition_features, feature_name, feature_value)

    return my_edition_features


def get_casted_feature_value(feature_value):
    if feature_value == "false":
        return False
    elif feature_value == "true":
        return True
    elif feature_value == "?":
        return 0
    else:
        try:
            return float(feature_value)
        except ValueError:
            return feature_value


def select_tree_path_to_apply(trained_model_name, edition_features):
    """"
    Method responsible for getting the tree path in the model that we are going to evaluate.
    As we train new models, we just have to include more if condition in this method. The specifications should be
        in each class that was built specifically for the trained_model.
    :param trained_model_name: the string with the name of the trained model (e.g. re_train_01, re_train_58Features).
    :param edition_features: the features of the edition, with the values that changed in this edition.
    :return: return the name of the model that contains the equation to classify a edition as vandalism or regular.
    """

    if trained_model_name == "ensemble_training_66_perc":
        return EnsembleTraining66Perc.predict(edition_features)
    else:
        return None


def apply_model_to_predict(model_weights, edition_features):
    model_weights_attributes = vars(model_weights)
    edition_features_attributes = vars(edition_features)
    equation_value = model_weights.COEFFICIENT  # starts by the coefficient, since it's not multiplied by any feature.

    for feature_name in edition_features_attributes:
        feature_name_in_model = feature_name
        if feature_name == "TIME_DOW":  # necessary step because this feature have a different name on the model
            feature_name_in_model = "TIME_DOW_" + str.upper(getattr(edition_features, feature_name))

        if feature_name_in_model in model_weights_attributes:
            feature_weight_result_aux = multiply_feature_model(getattr(edition_features, feature_name), getattr(model_weights, feature_name_in_model))
            equation_value += feature_weight_result_aux
            add_to_rank(feature_name_in_model, feature_weight_result_aux)

    PredictionResults.LOGISTIC_REGRESSION_EQUATION_VALUE = equation_value
    PredictionResults.PROBABILITY_OF_VANDALISM = calculate_probability_vandalism(equation_value)

    PredictionResults.FEATURE_RANKING.sort(key=RankAttribute.get_feature_name, reverse=True)
    calculate_k_means()

    return PredictionResults.PROBABILITY_OF_VANDALISM


def apply_model_to_predict_simplified(model_weights, edition_features):
    """
    In this version of the method, the k-means doesnt run, we are not interested in the features.
    :param model_weights: the model weights, used to calculate the estimate probability
    :param edition_features: the features of the edition.
    :return: the probability of vandalism.
    """

    model_weights_attributes = vars(model_weights)
    edition_features_attributes = vars(edition_features)
    equation_value = model_weights.COEFFICIENT  # starts by the coefficient, since it's not multiplied by any feature.

    for feature_name in edition_features_attributes:
        if feature_name in model_weights_attributes:
            feature_weight_result_aux = multiply_feature_model(getattr(edition_features, feature_name), getattr(model_weights, feature_name))
            equation_value += feature_weight_result_aux

    PredictionResults.LOGISTIC_REGRESSION_EQUATION_VALUE = equation_value
    PredictionResults.PROBABILITY_OF_VANDALISM = calculate_probability_vandalism(equation_value)
    return PredictionResults.PROBABILITY_OF_VANDALISM


def multiply_feature_model(edition_value, model_weight):
    try:
        if edition_value == "true":
            edition_value = 1
        elif edition_value == "false":
            edition_value = 0

        return edition_value * model_weight
    except:
        return 0


def calculate_probability_vandalism(equation_value):
    try:
        return 1 / (1 + math.exp(equation_value * -1))
    except OverflowError:  # TODO: better treatment of this exception is needed. Temporary solution.
        if equation_value < 0:  # In case of overflow, if the number is negative, then the probability of vandalism is low.
            return 0
        else:
            return 1


def add_to_rank(feature_name, feature_weight_value):
    #if feature_weight_value > 0:
    rank_aux = RankAttribute()
    rank_aux.feature_name = feature_name

    if feature_weight_value > LMTSettings.NORMALIZING_RANK_VALUE:
        feature_weight_value = LMTSettings.NORMALIZING_RANK_VALUE
    rank_aux.feature_weight_value = feature_weight_value

    PredictionResults.FEATURE_RANKING.append(rank_aux)


def calculate_k_means():
    model_weights = []

    if len(PredictionResults.FEATURE_RANKING) <= 1:
        return

    for ranking_aux in PredictionResults.FEATURE_RANKING:
        model_weights.append(ranking_aux.feature_weight_value)

    k_means_calculation = KMeans(n_clusters=2, init="random", n_init=10, max_iter=300)
    array_to_calculate = np.array(model_weights).reshape(-1, 1)
    k_means_calculation.fit(array_to_calculate)

    # k_means_calculation.cluster_centers_  --Used to get the centers found by KMeans

    important_group = k_means_calculation.labels_[0]

    for (group_aux, feature_rank_aux) in zip(k_means_calculation.labels_, PredictionResults.FEATURE_RANKING):
        feature_importance = FeatureImportance()
        feature_importance.feature_name = feature_rank_aux.feature_name
        if group_aux == important_group:
            feature_importance.is_important = True
            PredictionResults.VANDALISM_FEATURES.append(feature_importance.feature_name)
            PredictionResults.VANDALISM_FEATURES_VALUES.append(feature_rank_aux.feature_weight_value)

            feature_name_description = UtilExcel.get_feature_description(feature_importance.feature_name)
            PredictionResults.VANDALISM_FEATURES_DESCRIPTIONS.append(feature_name_description)

        PredictionResults.K_MEANS_LABELS.append(feature_importance)


def define_vandalism(vandalism_probability):
    if vandalism_probability > LMTSettings.VANDALISM_TRESHOLD:
        PredictionResults.IS_VANDALISM = True
        return True

    return False


def clear_prediction_values():
    PredictionResults.PROBABILITY_OF_VANDALISM = 0.0
    PredictionResults.IS_VANDALISM = False
    PredictionResults.LOGG_ODDS = 0.0
    PredictionResults.FEATURE_RANKING = []
    PredictionResults.K_MEANS_LABELS = []
    PredictionResults.VANDALISM_FEATURES = []
    PredictionResults.VANDALISM_FEATURES_VALUES = []
    PredictionResults.VANDALISM_FEATURES_DESCRIPTIONS = []
    PredictionResults.LOGISTIC_REGRESSION_EQUATION_VALUE = 0.0


def load_models_to_memory(number_of_models, models_folder_path, model_classify_regular_edition: False):
    all_models = {}

    for model_number in range(0, number_of_models):
        model_name = "model" + str(model_number+1)  # Add 1, because the model number count starts in 1.
        model_weights_file_path = models_folder_path + model_name + '.txt'
        model_weights = get_model_weights(model_weights_file_path, model_classify_regular_edition)

        all_models[model_name] = model_weights

    return all_models

