from model.randomforest import RandomForest
from modelling.chin import Chainer



def model_predict(data, chainer:Chainer =None):
    results = []
    print("RandomForest")
    model = RandomForest("RandomForest", data.get_embeddings(), data.get_type())
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data)
    model.print_results(data, chainer)
    model.print_results(data, chainer,1)
    model.print_results(data, chainer,2)


# def model_evaluate(model, data):
#     model.print_results(data)