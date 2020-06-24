import os

import keras
import keras.backend as backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import CSVLogger, History
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, LabelEncoder, label_binarize

"""
    Created by Mohsen Naghipourfar on 8/1/18.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""

n_epochs = 300
batch_size = 32


def create_regressor(n_features, layers, n_outputs, optimizer=None):
    input_layer = Input(shape=(n_features,))
    dense = Dense(layers[0], activation='relu', name="dense_0")(input_layer)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.5)(dense)

    for i, layer in enumerate(layers[1:]):
        dense = Dense(layer, activation='relu', name="dense_{0}".format(i + 1))(dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.5)(dense)
    dense = Dense(n_outputs, activation='sigmoid', name="output")(dense)
    model = Model(inputs=input_layer, outputs=dense)
    if optimizer is None:
        optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)
    model.compile(optimizer=optimizer, loss=["mse"], metrics=["mae"])
    return model


def random_classifier(drug_name=None, prediction_class=None):
    accuracies = {}
    data_directory = '../Data/CCLE/Classification/FS/'
    if drug_name:
        compounds = [drug_name + ".csv"]
    else:
        compounds = os.listdir(data_directory)
    print("All Compounds:")
    print(compounds)
    for compound in compounds:
        if compound.endswith(".csv") and not (
                compound.__contains__("PLX4720") or compound.__contains__("Panobinostat")):
            name = compound.split(".")[0]
            print("*" * 50)
            print(compound)
            print("Loading Data...")
            x_data, y_data = load_data(data_path=data_directory + compound, feature_selection=True)
            print("Data has been Loaded!")
            x_data = normalize_data(x_data)
            print("Data has been normalized!")

            n_samples = x_data.shape[0]
            if prediction_class is None:
                y_pred = np.random.random_integers(low=0, high=1, size=(n_samples, 1))
            else:
                if prediction_class == 1:
                    y_pred = np.ones(shape=[n_samples, 1])
                else:
                    y_pred = np.zeros(shape=[n_samples, 1])
            accuracies[name] = accuracy_score(y_data, y_pred)
            print("%s's Accuracy\t:\t%.4f%%" % (compound.split(".")[0], 100 * accuracy_score(y_data, y_pred)))

    log_path = "../Results/Classification/ML/"
    log_name = "Random" + "-" + str(prediction_class) + ".csv" if prediction_class is not None else "Random.csv"
    accuracies = pd.DataFrame(accuracies, index=[0])
    accuracies.to_csv(log_path + log_name)


def create_SAE(n_features=50000, n_code=12):
    input_layer = Input(shape=(n_features,))
    dense = Dense(2048, activation='relu', name="dense_0")(input_layer)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.2)(dense)

    dense = Dense(1024, activation='relu', name="dense_1")(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.5)(dense)
    dense = Dense(256, activation='relu', name="dense_2")(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.5)(dense)
    dense = Dense(64, activation='relu', name="dense_3")(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.5)(dense)
    encoded = Dense(n_code, activation='relu', name="encoded")(dense)

    dense = Dense(512, activation="relu", name="dense_4")(encoded)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.5)(dense)
    decoded = Dense(n_features, activation='sigmoid', name="decoded")(dense)
    cl_output = Dense(2, activation="softmax", name="classifier")(encoded)

    model = Model(inputs=input_layer, outputs=[decoded, cl_output])
    model.summary()
    lambda_value = 9.5581e-3

    def contractive_loss(y_pred, y_true):
        mse = backend.mean(backend.square(y_true - y_pred), axis=1)

        w = backend.variable(value=model.get_layer('encoded').get_weights()[0])  # N inputs N_hidden
        w = backend.transpose(w)  # N_hidden inputs N
        h = model.get_layer('encoded').output
        dh = h * (1 - h)  # N_batch inputs N_hidden

        # N_batch inputs N_hidden * N_hidden inputs 1 = N_batch inputs 1
        contractive = lambda_value * backend.sum(dh ** 2 * backend.sum(w ** 2, axis=1), axis=1)

        return mse + contractive

    reconstructor_loss = contractive_loss
    classifier_loss = "categorical_crossentropy"
    optimizer = keras.optimizers.Nadam(lr=0.005, beta_1=0.95)
    model.compile(optimizer=optimizer, loss=[reconstructor_loss, classifier_loss],
                  loss_weights=[0.005, 0.005],
                  metrics={"decoded": ["mae", "mse", "mape"], "classifier": "acc"})
    return model


def create_classifier(n_features=51, layers=None, n_outputs=1):
    if layers is None:
        layers = [1024, 256, 64, 16, 4]
    input_layer = Input(shape=(n_features,))
    dense = Dense(layers[0], activation='relu', name="dense_0")(input_layer)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.5)(dense)
    for i, layer in enumerate(layers[1:]):
        dense = Dense(layer, activation='relu', name="dense_{0}".format(i + 1))(dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.5)(dense)
    optimizer = keras.optimizers.adamax()
    if n_outputs > 1:
        dense = Dense(n_outputs, activation='softmax', name="output")(dense)
        loss = keras.losses.categorical_crossentropy
    else:
        dense = Dense(n_outputs, activation='sigmoid', name="output")(dense)
        loss = keras.losses.binary_crossentropy
    model = Model(inputs=input_layer, outputs=dense)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model


def load_data(data_path="../Data/CCLE/drug_response.csv", feature_selection=False):
    if data_path.__contains__("/FS/"):
        data = pd.read_csv(data_path)
    else:
        data = pd.read_csv(data_path, index_col="Cell Line")
    if data_path.__contains__("Regression"):
        y_data = data['IC50 (uM)']
        x_data = data.drop(['IC50 (uM)'], axis=1)
    else:
        y_data = data['class']
        x_data = data.drop(['class'], axis=1)
        label_encoder = LabelEncoder()
        y_data = label_encoder.fit_transform(y_data)
        y_data = np.reshape(y_data, (-1, 1))
        y_data = keras.utils.to_categorical(y_data, 2)
    if feature_selection and not data_path.__contains__("/FS/"):
        feature_names = list(pd.read_csv("../Data/BestFeatures.csv", header=None).loc[0, :])
        x_data = data[feature_names]
    return np.array(x_data), np.array(y_data)


def produce_classification_data(compounds):
    for compound in compounds:
        name = compound.split(".")[0]
        print(compound, end="\t")
        data = pd.read_csv("../Data/CCLE/Regression/" + name + "_preprocessed.csv")
        data['class'] = np.nan
        data.loc[data['IC50 (uM)'] >= 8, 'class'] = 1  # resistant
        data.loc[data['IC50 (uM)'] < 8] = 0  # sensitive
        data.dropna(how='any', axis=0, inplace=True)
        data.drop(["IC50 (uM)"], axis=1, inplace=True)
        data.to_csv("../Data/CCLE/Classification/" + name + ".csv", index_label="Cell Line")
        print("Finished!")


def normalize_data(x_data, y_data=None):
    x_data = pd.DataFrame(normalize(np.array(x_data), axis=0, norm='max')).values
    if y_data is not None:
        y_data = pd.DataFrame(np.reshape(np.array(y_data), (-1, 1)))
        y_data = pd.DataFrame(normalize(np.array(y_data), axis=0, norm='max'))
        return np.array(x_data), np.array(y_data)
    return np.array(x_data)


def regressor(drug_name=None):
    data_directory = '../Data/CCLE/Regression/'
    if drug_name:
        compounds = [drug_name + ".csv"]
    else:
        compounds = os.listdir(data_directory)
    print("All Compounds:")
    print(compounds)
    for compound in compounds:
        if compound.endswith("_preprocessed.csv"):
            print("*" * 50)
            print(compound)
            print("Loading Data...")
            x_data, y_data = load_data(data_path=data_directory + compound, feature_selection=True)
            print("Data has been Loaded!")
            x_data, y_data = normalize_data(x_data, y_data)
            print("Data has been normalized!")
            x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, shuffle=True)
            print("x_train shape\t:\t" + str(x_train.shape))
            print("y_train shape\t:\t" + str(y_train.shape))
            print("x_test shape\t:\t" + str(x_test.shape))
            print("y_test shape\t:\t" + str(y_test.shape))
            # for optimizer in optimizers:
            model = create_regressor(x_train.shape[1], [1024, 256, 64, 4], 1, None)
            logger_path = '../Results/Regression/' + compound.split(".")[0] + ".log"
            csv_logger = CSVLogger(logger_path)
            model.summary()
            model.fit(x=x_train,
                      y=y_train,
                      batch_size=batch_size,
                      epochs=n_epochs,
                      validation_data=(x_test, y_test),
                      verbose=2,
                      shuffle=True,
                      callbacks=[csv_logger])

            result = pd.read_csv(logger_path, delimiter=',')
            plt.figure(figsize=(15, 10))
            plt.plot(result['epoch'], result["loss"], label="Training Loss")
            plt.plot(result['epoch'], result["val_loss"], label="Validation Loss")
            plt.xlabel("Epochs")
            plt.ylabel("MSE Loss")
            plt.xticks([i for i in range(0, n_epochs + 5, 5)])
            plt.yticks(np.arange(0.25, -0.05, -0.05).tolist())
            plt.title(compound.split(".")[0])
            plt.grid()
            plt.savefig("../Results/Regression/images/%s.png" % compound.split(".")[0])
            plt.close("all")
            model.save("../Results/Regression/%s.h5" % compound.split(".")[0])


def regressor_with_different_optimizers():
    data_path = "../Data/CCLE/Regression/ZD-6474_preprocessed.csv"
    optimizers = [
        keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=1e-6, nesterov=True),
        keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True),
        keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-6, nesterov=True),
        keras.optimizers.Adagrad(lr=0.01, decay=1e-6),
        keras.optimizers.Adadelta(lr=1.0, rho=0.95, decay=1e-6),
        keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.99, decay=1e-6),
        keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999)
    ]
    print("Loading Data...")
    x_data, y_data = load_data(data_path, feature_selection=True)
    print("Data has been Loaded.")
    print("Normalizing Data...")
    x_data, y_data = normalize_data(x_data, y_data)
    print("Data has been normalized.")
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, shuffle=True)
    print("x_train shape\t:\t" + str(x_train.shape))
    print("y_train shape\t:\t" + str(y_train.shape))
    print("x_test shape\t:\t" + str(x_test.shape))
    print("y_test shape\t:\t" + str(y_test.shape))

    n_features = x_train.shape[1]
    layers = [1024, 256, 64, 8]
    n_outputs = 1

    for idx, optimizer in enumerate(optimizers):
        model = create_regressor(n_features, layers, n_outputs, optimizer)
        logger_path = "../Results/Optimizers/"
        optimizer_name = str(optimizer.__class__).split(".")[-1].split("\'")[0] + "_"
        optimizer_name += '_'.join(
            ["%s_%.4f" % (key, value) for (key, value) in optimizer.get_config().items()])
        optimizer_name += '.log'
        csv_logger = CSVLogger(logger_path + optimizer_name)
        model.summary()
        model.fit(x=x_train,
                  y=y_train,
                  batch_size=batch_size,
                  epochs=n_epochs,
                  validation_data=(x_test, y_test),
                  verbose=2,
                  shuffle=True,
                  callbacks=[csv_logger])


def regressor_with_k_best_features(k=50):
    data_directory = '../Data/CCLE/'
    compounds = os.listdir(data_directory)
    feature_names = list(pd.read_csv("../Data/BestFeatures.csv", header=None).loc[0, :])
    for compound in compounds:
        print("Loading Data...")
        x_data, y_data = load_data(data_path=data_directory + compound)
        print("Data has been Loaded!")
        x_data = x_data[feature_names]
        x_data, y_data = normalize_data(x_data, y_data)
        print("Data has been normalized!")
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, shuffle=True)
        print("x_train shape\t:\t" + str(x_train.shape))
        print("y_train shape\t:\t" + str(y_train.shape))
        print("x_test shape\t:\t" + str(x_test.shape))
        print("y_test shape\t:\t" + str(y_test.shape))

        for k in [50, 40, 30, 20, 10, 5, 4, 3, 2, 1]:
            model = create_regressor(x_train.shape[1], [32, 16, 4], 1)
            dir_name = "../Results/Drugs/%s/%dFeaturesSelection" % (compound.split(".")[0], k)
            os.makedirs(dir_name)
            csv_logger = CSVLogger(dir_name + '/best_%s_%d.log' % (compound.split(".")[0], k))
            model.fit(x=x_train,
                      y=y_train,
                      batch_size=batch_size,
                      epochs=n_epochs,
                      validation_data=(x_test, y_test),
                      verbose=2,
                      shuffle=True,
                      callbacks=[csv_logger])
            import csv
            with open("../Results/Drugs/%s/%s.csv" % (compound.split(".")[0], compound.split(".")[0]), 'a') as file:
                writer = csv.writer(file)
                loss = model.evaluate(x_test.as_matrix(), y_test.as_matrix(), verbose=0)
                loss.insert(0, k)
                writer.writerow(loss)
        df = pd.read_csv("../Results/Drugs/%s/%s.csv" % (compound.split(".")[0], compound.split(".")[0]), header=None)
        plt.figure()
        plt.plot(df[0], df[1], "-o")
        plt.xlabel("# of Features")
        plt.ylabel("Mean Absolute Error")
        plt.title(compound.split(".")[0])
        plt.savefig("../Results/Drugs/%s/%s.png" % (compound.split(".")[0], compound.split(".")[0]))


def classifier(drug_name=None):
    data_directory = '../Data/CCLE/Classification/FS/'
    if drug_name:
        compounds = [drug_name + ".csv"]
    else:
        compounds = os.listdir(data_directory)
    print("All Compounds:")
    print(compounds)
    for compound in compounds:
        if compound.endswith(".csv"):
            print("*" * 50)
            print(compound)
            print("Loading Data...")
            x_data, y_data = load_data(data_path=data_directory + compound, feature_selection=True)
            print("Data has been Loaded!")
            x_data = normalize_data(x_data)
            print("Data has been normalized!")
            # x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.05, shuffle=True)
            # print("x_train shape\t:\t" + str(x_train.shape))
            # print("y_train shape\t:\t" + str(y_train.shape))
            # print("x_test shape\t:\t" + str(x_test.shape))
            # print("y_test shape\t:\t" + str(y_test.shape))

            logger_path = "../Results/Classification/CV/"
            # plt.figure(figsize=(15, 10))
            # plt.title(compound.split(".")[0])
            model = None
            for k in range(10, 15, 5):
                model = KerasClassifier(build_fn=create_classifier,
                                        epochs=500,
                                        batch_size=64,
                                        verbose=2,
                                        )
                # y_data = encode_labels(y_data, 2)
                x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25)
                model.fit(x_train, y_train, validation_data=(x_test, y_test))

                print(x_test.shape)
                print(y_test.shape)

                y_pred = model.predict(x_test)
                y_pred = np.reshape(y_pred, (-1, 1))
                y_test = np.reshape(y_test, (-1, 1))
                print("Accuracy: %.4f %%" % (accuracy_score(y_test, y_pred) * 100))

                print(y_pred.shape)
                print(y_test.shape)
                n_classes = y_pred.shape[1]
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])

                # Compute micro-average ROC curve and ROC area
                fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                plt.close("all")
                plt.figure()
                lw = 2
                plt.plot(fpr[0], tpr[0], color='darkorange',
                         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
                plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver operating characteristic for %s' % compound.split(".")[0])
                plt.legend(loc="lower right")
                # plt.show()
                plt.savefig("../Results/Classification/ML/ROC/Deep Learning/%s.pdf" % (compound.split(".")[0]))

                # log_name = "Stratified %s-%d-cv.csv" % (compound.split(".")[0], k)
                # for x_train_cv, x_validation, y_train_cv, y_validation in stratified_kfold(x_data, y_data, k=k):
                #     label_encoder = LabelEncoder()
                #     y_train_cv = label_encoder.fit_transform(y_train_cv)
                #     y_train_cv = np.reshape(y_train_cv, (-1, 1))
                #     y_train_cv = keras.utils.to_categorical(y_train_cv, 2)
                #
                #     y_validation = label_encoder.transform(y_validation)
                #     y_validation = np.reshape(y_validation, (-1, 1))
                #     y_validation = keras.utils.to_categorical(y_validation, 2)
                #     model.fit(x=x_train_cv,
                #               y=y_train_cv,
                #               batch_size=batch_size,
                #               epochs=n_epochs,
                #               validation_data=(x_validation, y_validation),
                #               verbose=0,
                #               shuffle=True)
                #     score = model.evaluate(x_validation, y_validation, verbose=0)
                #     print("Stratified %d-fold %s %s: %.2f%%" % (
                #         k, compound.split(".")[0], model.metrics_names[1], score[1] * 100))
                #     cross_validation_scores.append(score[1] * 100)
                # model.save(filepath="../Results/Classification/%s.h5" % compound.split(".")[0])
                # np.savetxt(fname=logger_path + log_name, X=np.array(cross_validation_scores), delimiter=',')
                # plt.plot(cross_validation_scores, label="%d-fold cross validation")

            # result = pd.read_csv(logger_path, delimiter=',')
            # plt.xlabel("Folds")
            # plt.ylabel("Accuracy")
            # plt.xticks([i for i in range(0, n_epochs + 5, 5)], rotation=90)
            # plt.yticks(np.arange(0, 1.05, 0.05).tolist())
            # plt.title(compound.split(".")[0])
            # plt.grid()
            # plt.legend(loc="upper right")
            # plt.savefig("../Results/Classification/images/%s.png" % compound.split(".")[0])
            # plt.close("all")
    print("Finished!")


def encode_labels(y_data, n_classes=2):
    label_encoder = LabelEncoder()
    y_data = label_encoder.fit_transform(y_data)
    y_data = np.reshape(y_data, (-1, 1))
    y_data = keras.utils.to_categorical(y_data, n_classes)
    return y_data


def plot_results(path="../Results/Classification/"):
    logs = os.listdir(path)
    print(logs)
    for log in logs:
        if os.path.isfile(path + log) and log.endswith(".log"):
            result = pd.read_csv(path + log, delimiter=',')
            plt.figure(figsize=(15, 10))
            plt.plot(result['epoch'], result["val_acc"])
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.title(log.split(".")[0])
            plt.savefig("../Results/Classification/images/%s.png" % log.split(".")[0])
            plt.close("all")


def plot_roc_curve(path="../Results/Classification/"):
    models = os.listdir(path)
    models = [models[i] for i in range(len(models)) if models[i].endswith(".h5")]
    for model in models:
        drug_name = model.split(".")[0]
        # print(drug_name + "\t:\t", end="")
        model = keras.models.load_model(path + model)
        x_data, y_data = load_data(data_path='../Data/CCLE/Classification/' + drug_name + ".csv",
                                   feature_selection=True)
        y_pred = model.predict(x_data.as_matrix())

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(2):
            fpr[i], tpr[i], _ = roc_curve(y_data[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure()
        lw = 2
        plt.plot(fpr[0], tpr[0], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()


def machine_learning_classifiers(drug_name=None, alg_name="SVM"):
    data_directory = '../Data/CCLE/Classification/FS/'
    if not drug_name:
        compounds = os.listdir(data_directory)
    else:
        compounds = [drug_name + ".csv"]
    log_path = "../Results/Classification/ML/"
    accuracies = {}
    for k in range(5, 15, 5):
        log_name = alg_name + "-Stratified-%d-cv.csv" % k
        for compound in compounds:
            if compound.endswith(".csv") and not (
                    compound.__contains__("PLX4720") or compound.__contains__("Panobinostat")):
                name = compound.split(".")[0]
                print("*" * 50)
                print(compound)
                print("Loading Data...")
                x_data, y_data = load_data(data_path=data_directory + compound, feature_selection=True)
                print("Data has been Loaded!")
                x_data = normalize_data(x_data)
                print("Data has been normalized!")

                accuracies[name] = 0.0
                ml_classifier = OneVsRestClassifier(svm.SVC(C=1.0, kernel='rbf', probability=True))
                if alg_name == "SVM":
                    ml_classifier = OneVsRestClassifier(svm.SVC(C=1.0, kernel='rbf', probability=True))
                elif alg_name == "RandomForest":
                    ml_classifier = OneVsRestClassifier(RandomForestClassifier())
                elif alg_name == "GradientBoosting":
                    ml_classifier = OneVsRestClassifier(GradientBoostingClassifier(learning_rate=0.01))

                y_data = label_binarize(y_data, classes=[0, 1])
                print(y_data.shape)
                n_classes = y_data.shape[1]
                # x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25)

                for x_train_cv, x_validation, y_train_cv, y_validation in stratified_kfold(x_data, y_data, k=k):
                    ml_classifier = ml_classifier.fit(x_train_cv, y_train_cv)
                    y_pred = ml_classifier.predict(x_validation)
                    accuracies[name] += accuracy_score(y_validation, y_pred)
                    print(name, k, accuracy_score(y_validation, y_pred) * 100)
                # ml_classifier.fit(x_train, y_train)

                # y_pred = ml_classifier.predict(x_test)
                # y_pred = np.reshape(y_pred, (-1, 1))
                # target_names = ["class_0: Resistant", "class_1: Sensitive"]
                # string = classification_report(y_test, y_pred, target_names=target_names)
                # with open("../Results/Classification/ML/Classification Report/%s/%s.txt" % (alg_name, name), "w") as f:
                #     f.write(string)
                # fpr = dict()
                # tpr = dict()
                # roc_auc = dict()
                # for i in range(n_classes):
                #     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
                #     roc_auc[i] = auc(fpr[i], tpr[i])
                #
                # # Compute micro-average ROC curve and ROC area
                # fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
                # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                # plt.close("all")
                # plt.figure()
                # lw = 2
                # plt.plot(fpr[0], tpr[0], color='darkorange',
                #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
                # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                # plt.xlim([0.0, 1.0])
                # plt.ylim([0.0, 1.05])
                # plt.xlabel('False Positive Rate')
                # plt.ylabel('True Positive Rate')
                # plt.title('Receiver operating characteristic for %s' % name)
                # plt.legend(loc="lower right")
                # plt.show()
                # plt.savefig("../Results/Classification/ML/ROC/%s/%s.pdf" % (alg_name, name))
                accuracies[name] /= k
                print("Mean Accuracy = %.4f" % accuracies[name])

        results = pd.DataFrame(data=accuracies, index=[0])
        results.to_csv(log_path + log_name)
    print("Finished!")


def kfold_cross_validation(x_data, y_data, k=10):
    kf = KFold(n_splits=k, shuffle=True)
    for train_idx, test_idx in kf.split(x_data):
        train_idx = list(train_idx)
        test_idx = list(test_idx)
        x_train, x_test = x_data[train_idx], x_data[test_idx]
        y_train, y_test = y_data[train_idx], y_data[test_idx]
        yield x_train, x_test, y_train, y_test
    data_directory = '../Data/CCLE/Classification/'
    compounds = os.listdir(data_directory)
    log_path = "../Results/Classification/ML/"


def stratified_kfold(x_data, y_data, k=10):
    skfold = StratifiedKFold(n_splits=k, shuffle=True)
    for train_idx, test_idx in skfold.split(x_data, y_data):
        train_idx = list(train_idx)
        test_idx = list(test_idx)
        x_train, x_test = x_data[train_idx], x_data[test_idx]
        y_train, y_test = y_data[train_idx], y_data[test_idx]
        yield x_train, x_test, y_train, y_test


def generate_small_datas():
    data_directory = '../Data/CCLE/Classification/'
    compounds = ["AEW541.csv"]
    print("All Compounds:")
    print(compounds)
    for compound in compounds:
        if compound.endswith(".csv"):
            print(compound, end="\t")
            x_data, y_data = load_data(data_directory + compound, feature_selection=True)
            data = pd.DataFrame(x_data)
            data['class'] = y_data
            data.to_csv(data_directory + "FS/" + compound.split(".")[0] + ".csv")
            print("Finished!")


def generate_latex_table_for_data_description(compounds):
    for compound in compounds:
        data = pd.read_csv("../Data/CCLE/Classification/" + compound)
        name = compound.split(".")[0]
        # threshold = float(thresholds[thresholds[0] == name][1])
        n_resistant = len(data[data['class'] == 1])
        n_sensitive = len(data[data['class'] == 0])
        n_total = n_resistant + n_sensitive
        p_resistant = float(n_resistant / n_total) * 100.0
        p_sensitive = float(n_sensitive / n_total) * 100.0
        print("%s &  %d & %.2f" % (
            compound.split(".")[0], n_resistant, p_resistant) + "\\%" + " & %d & %.2f" % (
                  n_sensitive, p_sensitive) + "\\% \\\\")


def repr_learner(drug_name=None):
    data_directory = "../Data/CCLE/Classification/"
    compounds = [drug_name + ".csv"]
    print("All Compounds:")
    print(compounds)
    for compound in compounds:
        if compound.endswith(".csv"):
            print("*" * 50)
            print(compound)
            drug_name = compound.split(".")[0]
            print("Loading Data...")
            x_data, y_data = load_data(data_path=data_directory + compound, feature_selection=True)
            print("Data has been Loaded!")
            x_data = normalize_data(x_data)
            print("Data has been normalized!")
            x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.35, stratify=y_data,
                                                                shuffle=True)
            print("x_train shape\t:\t" + str(x_train.shape))
            print("y_train shape\t:\t" + str(y_train.shape))
            print("x_test shape\t:\t" + str(x_test.shape))
            print("y_test shape\t:\t" + str(y_test.shape))

            model = create_SAE(x_data.shape[1], n_code=12)
            os.makedirs("../Results/logs/", exist_ok=True)
            os.makedirs("../Results/models/", exist_ok=True)
            os.makedirs("../Results/SAE Data/", exist_ok=True)
            csv_logger = CSVLogger("../Results/logs/SAE.log")
            history = History()

            model.fit(x_train,
                      [x_train, y_train],
                      epochs=1000,
                      batch_size=32,
                      validation_data=(x_test, [x_test, y_test]),
                      callbacks=[csv_logger, history],
                      verbose=2)
            print(history.history.keys())
            layer_name = "encoded"
            model.save(filepath="../Results/models/" + "SAE.h5")
            encoded_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
            encoded_output = pd.DataFrame(encoded_layer_model.predict(x_data))
            encoded_output = pd.concat([encoded_output, pd.DataFrame(y_data)], axis=1)
            encoded_output.to_csv("../Results/SAE Data/" + drug_name + "encoded_SAE.csv")


if __name__ == '__main__':
    # generate_small_datas()
    # random_classifier(None)
    # random_classifier(None, 0)
    # random_classifier(None, 1)
    # machine_learning_classifiers(None, "SVM")
    # machine_learning_classifiers(None, "RandomForest")
    # machine_learning_classifiers(None, "GradientBoosting")
    # classifier("AEW541")
    repr_learner("AEW541")
