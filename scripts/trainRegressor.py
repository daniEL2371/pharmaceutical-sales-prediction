from IPython.display import Markdown, display, Image

from sklearn.ensemble import RandomForestRegressor

import seaborn as sns
import matplotlib.pyplot as plt

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


def save_fig(fig, path="../random-forest-result.png"):
    fig.savefig(path)


def plot_importance(random_feat_imp):
    fig = plt.figure(figsize=(9, 7))
    sns.barplot(data=random_feat_imp, x="importance", y="features")
    plt.title("Feature importance", size=18)
    plt.xticks(rotation=60, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('features', fontsize=16)
    plt.ylabel("features importance", fontsize=16)
    plt.show()

    return fig


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
    save_fig(fig)

    random_feat_imp = rossPipeLine.get_feature_importance(model).sort_values(
        by=["importance"], ascending=False)

    fig = plot_importance(random_feat_imp)
    save_fig(fig, "../random-forest-feat-importance.png")
