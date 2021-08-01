from IPython.display import Markdown, display, Image

from sklearn.ensemble import RandomForestRegressor

from rossmanModelPipeline import RossmanModelPipeline
from cleanStoreDf import CleanStoreDf
from cleanTrainData import CleanTrainData
from preprocessRossmanData import PreprocessRossmanData
from helper import Helper


helper = Helper()


def save_metrics(score, loss, path="../random-forest-metrics.txt"):
    try:
        fileObj = open(path, "w")
        metrics = [f"R2 Score:   {score}\n", f"MAE loss: {loss}"]
        fileObj.writelines(metrics)
        fileObj.close()
    except:
        fileObj.close()


def save_test_fig(fig, path="../random-forest-result.png"):
    fig.savefig(path)


if __name__ == "__main__":
    train_df = helper.read_csv("../data/train.csv")
    store_df = helper.read_csv("../data/store.csv")
    test_df = helper.read_csv("../data/test.csv")

    clean_train_df = CleanTrainData().get_cleaned(train_df)
    clean_store_df = CleanStoreDf().get_cleaned(store_df)
    cleaned_rossman_data = PreprocessRossmanData().process(
        clean_train_df, clean_store_df)

    regressor = RandomForestRegressor(n_jobs=-1, max_depth=15, n_estimators=15)

    rossPipeLine = RossmanModelPipeline(cleaned_rossman_data, "RandomForset-2")
    pipeline, model = rossPipeLine.train(regressor=regressor)

    score, loss, res_df = rossPipeLine.test(model)
    display(res_df)
    save_metrics(score=score, loss=loss)
    fig = rossPipeLine.pred_graph(res_df)
    save_test_fig(fig)
