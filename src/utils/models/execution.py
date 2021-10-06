import pandas as pd

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import tensorflow as tf


def random_forest(training_set, test_set, k, timestamp, whole=False):
    X_train, y_train = training_set[training_set.columns[:-1]].values, training_set['label'].values
    X_test, y_test = test_set[training_set.columns[:-1]].values, test_set['label'].values

    k_fold = KFold(n_splits=k)

    results = pd.DataFrame(columns=['model', 'test_loss', 'test_accuracy'])
    results_path = 'src/results/model_results_' + timestamp + '.csv'
    results.to_csv(results_path)

    n_estimators = [500, 1000]
    max_depths = [15, 25, 50]

    if not whole:
        for k, (training_index, val_index) in enumerate(k_fold.split(X_train)):
            k_X_train, k_y_train = X_train[training_index], y_train[training_index]
            k_X_val, k_y_val = X_train[val_index], y_train[val_index]
            for n in n_estimators:
                for depth in max_depths:
                    print((n, depth))
                    cu_clf = RandomForestClassifier(random_state=1, n_estimators=n, max_depth=depth)
                    cu_clf.fit(k_X_train, k_y_train)
                    k_y_pred = cu_clf.predict(k_X_val)
                    results = pd.read_csv(results_path)
                    results.drop(results.columns[0], inplace=True, axis=1)
                    results.loc[len(results)] = ['rfc_' + str(n) + '_' + str(depth), '/', accuracy_score(k_y_val,
                                                                                                         k_y_pred)]
                    results.to_csv(results_path)
    else:
        for n in n_estimators:
            for depth in max_depths:
                print((n, depth))
                cu_clf = RandomForestClassifier(random_state=1, n_estimators=n, max_depth=depth)
                cu_clf.fit(X_train, y_train)
                y_pred = cu_clf.predict(X_test)
                results = pd.read_csv(results_path)
                results.drop(results.columns[0], inplace=True, axis=1)
                results.loc[len(results)] = ['rfc_' + str(n) + '_' + str(depth), '/', accuracy_score(y_test, y_pred)]
                results.to_csv(results_path)


def neural_network(training_set, test_set, k, models, timestamp, whole=False):
    X_train, y_train = training_set[training_set.columns[:-1]].values, training_set['label'].values
    X_test, y_test = test_set[training_set.columns[:-1]].values, test_set['label'].values

    k_fold = KFold(n_splits=k)

    results = pd.DataFrame(columns=['model', 'test_loss', 'test_accuracy'])
    results_path = 'src/results/model_results_' + timestamp + '.csv'
    results.to_csv(results_path)

    if not whole:
        for k, (training_index, val_index) in enumerate(k_fold.split(X_train)):
            k_X_train, k_y_train = X_train[training_index], y_train[training_index]
            k_X_val, k_y_val = X_train[val_index], y_train[val_index]
            for model in models:
                name = model[0]
                model = tf.keras.Sequential(model[1])
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
                model.fit(k_X_train, k_y_train, validation_data=(k_X_val, k_y_val), epochs=50, batch_size=64,
                          verbose=0, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15)])
                test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=X_test.shape[0])
                results = pd.read_csv(results_path)
                results.drop(results.columns[0], inplace=True, axis=1)
                results.loc[len(results)] = [name, test_loss, test_acc]
                results.to_csv(results_path)
    else:
        for model in models:
            name = model[0]
            model = tf.keras.Sequential(model[1])
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
            model.fit(X_train, y_train, epochs=50, batch_size=64,
                      verbose=0, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15)])
            test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=X_test.shape[0])
            results = pd.read_csv(results_path)
            results.drop(results.columns[0], inplace=True, axis=1)
            results.loc[len(results)] = [name, test_loss, test_acc]
            results.to_csv(results_path)
