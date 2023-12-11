from classification_utils import train_model_simple, test_model


class ManualEnsemble:
    """
    A class used to represent an ensemble of models for classification.

    Attributes
    ----------
    classifiers_df: list
        a list of tuples of the form (classifier, df)
    target : str
        the target variable

    Methods
    -------
    train(year):
        Trains each model in the ensemble.
    test(year):
        Tests each model in the ensemble and returns the summed probabilities from all models.
    """

    def __init__(self, classifiers_df, target):
        self.classifiers_df = classifiers_df
        self.target = target

    def train(self, year):
        for classifier, df in self.classifiers_df:
            train_model_simple(classifier, df, year, self.target)

    def test(self, year):
        y_test_prob = []
        y_test_gt = []
        conf_test = []

        y_train_prob = []
        y_train_gt = []
        conf_train = []

        y_test_tmID = []

        for classifier, df in self.classifiers_df:
            (
                y_test_gt_i,
                y_test_prob_i,
                conf_test_i,
                y_train_gt_i,
                y_train_prob_i,
                conf_train_i,
                y_test_tmID_i,
            ) = test_model(classifier, df, year, self.target)
            y_test_prob.append(y_test_prob_i)
            y_train_prob.append(y_train_prob_i)
            
            if len(conf_test) == 0: # First iteration
                conf_test = conf_test_i
                conf_train = conf_train_i
                y_test_gt = y_test_gt_i
                y_train_gt = y_train_gt_i
                y_test_tmID = y_test_tmID_i
            
            # To make sure that the order of the teams is the same for all models
            assert conf_test.equals(conf_test_i)
            assert conf_train.equals(conf_train_i)
            assert y_test_gt.equals(y_test_gt_i)
            assert y_train_gt.equals(y_train_gt_i)
            assert y_test_tmID.equals(y_test_tmID_i)
            
        y_test_prob = [sum(x) for x in zip(*y_test_prob)]
        y_train_prob = [sum(x) for x in zip(*y_train_prob)]

        return (
            y_test_gt,
            y_test_prob,
            conf_test,
            y_train_gt,
            y_train_prob,
            conf_train,
            y_test_tmID,
        )
