class BaseModel(object):

    def __init__(self):
        self.model = None

    def train(self, dataset):
        pass

    def save(self, output_path):
        pass

    def load(self, model_path):
        pass

    def vectorize(self, document):
        pass

