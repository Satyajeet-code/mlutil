# Collection of Useful Machine Learning workflow methods, and viz tools.
# such as grid search, full viz,specific plotting mechanisms etc.
#

# For Grid Search
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# For plotting
import matplotlib.pyplot as plt


def hyp_pipeline(self, X_train, X_test, y_train, y_test,
                       model, param_grid, cv=10, scoring_fit='neg_mean_squared_error',
                       do_probabilities=False, search_mode='GridSearchCV', n_iterations=0, n_jobs=-1):
        '''Hyper Parameter searching with GridSearch and RandomSearch'''

        fitted_model = None
    
        if(search_mode == 'GridSearchCV'):
            gs = GridSearchCV(
                estimator=model,
                param_grid=param_grid, 
                cv=cv, 
                n_jobs=n_jobs, 
                scoring=scoring_fit,
                verbose=1)
            fitted_model = gs.fit(X_train, y_train)
        elif(search_mode == 'RandomizedSearchCV'):
            rs = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid, 
                cv=cv,
                n_iter=n_iterations,
                n_jobs=n_jobs, 
                scoring=scoring_fit,
                verbose=1)
            fitted_model = rs.fit(X_train, y_train)
    
    
        if(fitted_model != None):
            if do_probabilities:
                pred = fitted_model.predict_proba(X_test)
            else:
                pred = fitted_model.predict(X_test)
            
        return fitted_model, pred

def plot_history(self,history):
        ''' Plot accuracy and loss of keras Models '''
        loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
        val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
        acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
        val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
        if len(loss_list) == 0:
            print('Loss is missing in history')
            return 
    
         # As loss always exists
        epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    # Loss
        plt.figure(1)
        for l in loss_list:
            plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
        for l in val_loss_list:
            plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
    
    # Accuracy
        plt.figure(2)
        for l in acc_list:
            plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
        for l in val_acc_list:    
            plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
