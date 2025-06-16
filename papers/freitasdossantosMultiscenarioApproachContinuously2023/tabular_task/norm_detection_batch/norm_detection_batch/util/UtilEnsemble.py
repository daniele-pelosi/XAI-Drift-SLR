from norm_detection_batch.lmt import PredictionResults, PredictionResultsEnsemble
from norm_detection_batch.lmt.RankAttribute import RankAttribute
from norm_detection_batch.util import Util


def apply_model_to_predict(model_weights, edition_features, current_predictor):
    clear_prediction_results()

    model_weights_attributes = vars(model_weights)
    edition_features_attributes = vars(edition_features)
    equation_value = model_weights.COEFFICIENT  # starts by the coefficient, since it's not multiplied by any feature.

    for feature_name in edition_features_attributes:
        feature_name_in_model = feature_name
        if feature_name == "TIME_DOW":  # necessary step because this feature have a different name on the model
            feature_name_in_model = "TIME_DOW_" + str.upper(getattr(edition_features, feature_name))

        if feature_name_in_model in model_weights_attributes:
            feature_weight_result_aux = Util.multiply_feature_model(getattr(edition_features, feature_name), getattr(model_weights, feature_name_in_model))
            equation_value += feature_weight_result_aux
            Util.add_to_rank(feature_name_in_model, feature_weight_result_aux)

    PredictionResults.LOGISTIC_REGRESSION_EQUATION_VALUE = equation_value
    PredictionResults.PROBABILITY_OF_VANDALISM = Util.calculate_probability_vandalism(equation_value)

    PredictionResults.FEATURE_RANKING.sort(key=RankAttribute.get_feature_name, reverse=True)
    Util.calculate_k_means()

    # Adding the important features found by the current predictor
    PredictionResultsEnsemble.VANDALISM_FEATURES_PER_PREDICTOR[current_predictor] = PredictionResults.VANDALISM_FEATURES
    PredictionResultsEnsemble.FEATURE_RANKING_PER_PREDICTOR[current_predictor] = PredictionResults.FEATURE_RANKING
    PredictionResultsEnsemble.EQUATION_VALUE_PER_PREDICTOR[current_predictor] = PredictionResults.LOGISTIC_REGRESSION_EQUATION_VALUE

    return PredictionResults.PROBABILITY_OF_VANDALISM


def clear_prediction_results():
    PredictionResults.FEATURE_RANKING = []
    PredictionResults.K_MEANS_LABELS = []
    PredictionResults.VANDALISM_FEATURES = []
    PredictionResults.VANDALISM_FEATURES_VALUES = []
    PredictionResults.VANDALISM_FEATURES_DESCRIPTIONS = []


def clear_prediction_ensemble_results():
    PredictionResultsEnsemble.VANDALISM_FEATURES_PER_PREDICTOR = {}
