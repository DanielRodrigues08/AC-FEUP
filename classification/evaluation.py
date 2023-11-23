from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, \
    roc_curve
import seaborn as sns
import matplotlib.pyplot as plt


def confusion_matrix_plot(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt='g',
        xticklabels=['No', 'Yes'],
        yticklabels=['No', 'Yes']
    )
    plt.ylabel("Prediction", fontsize=13)
    plt.xlabel("Actual", fontsize=13)
    plt.title("Confusion Matrix", fontsize=14)
    plt.show()


def roc_curve_plot(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC Curve')
    plt.show()


def model_score(y_test, y_pred, y_prob, plot=True):
    print("Prob: ", y_prob)
    print("Pred: ", y_pred)
    if plot:
        confusion_matrix_plot(y_test, y_pred)
        roc_curve_plot(y_test, y_prob)

    print(f"AUC: {round(roc_auc_score(y_test, y_prob) * 100, 2)}%")
    print(f"Accuracy: {round(accuracy_score(y_test, y_pred) * 100, 2)}%")
    print(f"Precision: {round(precision_score(y_test, y_pred) * 100, 2)}%")
    print(f"Recall: {round(recall_score(y_test, y_pred) * 100, 2)}%")
    print(f"F1: {round(f1_score(y_test, y_pred) * 100, 2)}%")


def plot_learning_curve(years, train_scores, test_scores, fig_size=(8, 6)):
    plt.figure()
    plt.plot(years, train_scores, marker='o', label='Training score', color='blue')
    plt.plot(years, test_scores, marker='o', label='Test score', color='red')

    plt.title('Learning Curve')
    plt.xlabel('Years')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
