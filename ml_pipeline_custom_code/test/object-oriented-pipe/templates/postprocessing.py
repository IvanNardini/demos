from sklearn.metrics import classification_report

class PostProcessing:

    def evaluate_classification(self, model, X_train, y_train, X_test, y_test):
        '''
        Evaluate classification
        params: model, X_train, y_train, X_test, y_test
        returns: None
        '''
        #Evaluate Train Sample
        # predictions_train = model.predict(X_train)
        # score_train = round(model.score(X_train, y_train), 2)
        # classification_train = classification_report(y_train, predictions_train)
        # print()
        # print('score: {}'.format(score_train))
        # print()
        # print('Classification report - Training')
        # print(classification_train)

        #Evaluate Test Sample
        predictions_test = model.predict(X_test)
        score_test = round(model.score(X_test, y_test), 3)
        classification_test = classification_report(y_test, predictions_test)
        print()
        print('score: {}'.format(score_test))
        print()
        print('Classification report - Test')
        print(classification_test)
        return None