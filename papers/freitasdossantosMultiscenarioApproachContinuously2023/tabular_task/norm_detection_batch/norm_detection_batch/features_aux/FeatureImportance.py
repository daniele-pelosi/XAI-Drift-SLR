class FeatureImportance:
    feature_name = ""
    is_important = False

    def __repr__(self):
        return "Feature: " + self.feature_name + " Important: " + str(self.is_important) + ""

    def __str__(self):
        return "Feature: " + self.feature_name + " Important: " + str(self.is_important) + ""
