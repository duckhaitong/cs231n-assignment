import random
import numpy as np
import matplotlib.pyplot as plt

from builtins import range
from builtins import object
from cs231n.data_utils import load_CIFAR10
from past.builtins import xrange


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):

        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
 
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
 
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
 
                pass

        return dists

    def compute_distances_one_loop(self, X):
 
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):

            pass

        return dists

    def compute_distances_no_loops(self, X):
      
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))


        pass

        return dists

    def predict_labels(self, dists, k=1):
       
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):

            closest_y = []
            
            pass

            pass


        return y_pred

if __name__ == '__main__':
    # Load the raw CIFAR-10 data.
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    # Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
    try:
       del X_train, y_train
       del X_test, y_test
       print('Clear previously loaded data.')
    except:
       pass

    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # As a sanity check, we print out the size of the training and test data.
    print('Training data shape: ', X_train.shape)
    print('Training labels shape: ', y_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)

    # # Visualize some examples from the dataset.
    # # We show a few examples of training images from each class.
    # classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # samples_per_class = 7
    # for y, cls in enumerate(classes):
    #     idxs = np.flatnonzero(y_train == y)
    #     idxs = np.random.choice(idxs, samples_per_class, replace=False)
    #     plt.figure("Example")
    #     for i, idx in enumerate(idxs):
    #         plt_idx = i * len(classes) + y + 1
    #         plt.subplot(samples_per_class, len(classes), plt_idx)
    #         plt.imshow(X_train[idx].astype('uint8'))
    #         plt.axis('off')
    #         if i == 0:
    #             plt.title(cls)

    # plt.show()

    # Subsample the data for more efficient code execution 
    num_train = 5000
    mask = list(range(num_train))
    X_train = X_train[mask]
    y_train = y_train[mask]

    num_test = 500
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Reshape the image data into rows (5000, 32 * 32 * 3) (500, 32 * 32 * 3) 
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    classifier = KNearestNeighbor()
    classifier.train(X_train, y_train)

    dists = classifier.compute_distances_two_loops(X_test)
    print(dists.shape)

    # Visualize the distance matrix: each row is a single test example and
    # its distances to training examples
    plt.imshow(dists, interpolation='none')
    plt.show()

    y_test_pred = classifier.predict_labels(dists, k=5)

    # Compute and print the fraction of correctly predicted examples
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / num_test
    print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))


# # You should expect to see approximately `27%` accuracy. Now lets try out a larger `k`, say `k = 5`:

# # In[ ]:


# y_test_pred = classifier.predict_labels(dists, k=5)
# num_correct = np.sum(y_test_pred == y_test)
# accuracy = float(num_correct) / num_test
# print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))


# # You should expect to see a slightly better performance than with `k = 1`.

# # **Inline Question 2**
# # 
# # We can also use other distance metrics such as L1 distance.
# # For pixel values $p_{ij}^{(k)}$ at location $(i,j)$ of some image $I_k$, 
# # 
# # the mean $\mu$ across all pixels over all images is $$\mu=\frac{1}{nhw}\sum_{k=1}^n\sum_{i=1}^{h}\sum_{j=1}^{w}p_{ij}^{(k)}$$
# # And the pixel-wise mean $\mu_{ij}$ across all images is 
# # $$\mu_{ij}=\frac{1}{n}\sum_{k=1}^np_{ij}^{(k)}.$$
# # The general standard deviation $\sigma$ and pixel-wise standard deviation $\sigma_{ij}$ is defined similarly.
# # 
# # Which of the following preprocessing steps will not change the performance of a Nearest Neighbor classifier that uses L1 distance? Select all that apply.
# # 1. Subtracting the mean $\mu$ ($\tilde{p}_{ij}^{(k)}=p_{ij}^{(k)}-\mu$.)
# # 2. Subtracting the per pixel mean $\mu_{ij}$  ($\tilde{p}_{ij}^{(k)}=p_{ij}^{(k)}-\mu_{ij}$.)
# # 3. Subtracting the mean $\mu$ and dividing by the standard deviation $\sigma$.
# # 4. Subtracting the pixel-wise mean $\mu_{ij}$ and dividing by the pixel-wise standard deviation $\sigma_{ij}$.
# # 5. Rotating the coordinate axes of the data.
# # 
# # $\color{blue}{\textit Your Answer:}$
# # 
# # 
# # $\color{blue}{\textit Your Explanation:}$
# # 

# # In[1]:


# # Now lets speed up distance matrix computation by using partial vectorization
# # with one loop. Implement the function compute_distances_one_loop and run the
# # code below:
# dists_one = classifier.compute_distances_one_loop(X_test)

# # To ensure that our vectorized implementation is correct, we make sure that it
# # agrees with the naive implementation. There are many ways to decide whether
# # two matrices are similar; one of the simplest is the Frobenius norm. In case
# # you haven't seen it before, the Frobenius norm of two matrices is the square
# # root of the squared sum of differences of all elements; in other words, reshape
# # the matrices into vectors and compute the Euclidean distance between them.
# difference = np.linalg.norm(dists - dists_one, ord='fro')
# print('One loop difference was: %f' % (difference, ))
# if difference < 0.001:
#     print('Good! The distance matrices are the same')
# else:
#     print('Uh-oh! The distance matrices are different')


# # In[ ]:


# # Now implement the fully vectorized version inside compute_distances_no_loops
# # and run the code
# dists_two = classifier.compute_distances_no_loops(X_test)

# # check that the distance matrix agrees with the one we computed before:
# difference = np.linalg.norm(dists - dists_two, ord='fro')
# print('No loop difference was: %f' % (difference, ))
# if difference < 0.001:
#     print('Good! The distance matrices are the same')
# else:
#     print('Uh-oh! The distance matrices are different')


# # In[ ]:


# # Let's compare how fast the implementations are
# def time_function(f, *args):
#     """
#     Call a function f with args and return the time (in seconds) that it took to execute.
#     """
#     import time
#     tic = time.time()
#     f(*args)
#     toc = time.time()
#     return toc - tic

# two_loop_time = time_function(classifier.compute_distances_two_loops, X_test)
# print('Two loop version took %f seconds' % two_loop_time)

# one_loop_time = time_function(classifier.compute_distances_one_loop, X_test)
# print('One loop version took %f seconds' % one_loop_time)

# no_loop_time = time_function(classifier.compute_distances_no_loops, X_test)
# print('No loop version took %f seconds' % no_loop_time)

# # You should see significantly faster performance with the fully vectorized implementation!

# # NOTE: depending on what machine you're using, 
# # you might not see a speedup when you go from two loops to one loop, 
# # and might even see a slow-down.


# # ### Cross-validation
# # 
# # We have implemented the k-Nearest Neighbor classifier but we set the value k = 5 arbitrarily. We will now determine the best value of this hyperparameter with cross-validation.

# # In[ ]:


# num_folds = 5
# k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

# X_train_folds = []
# y_train_folds = []
# ################################################################################
# # TODO:                                                                        #
# # Split up the training data into folds. After splitting, X_train_folds and    #
# # y_train_folds should each be lists of length num_folds, where                #
# # y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
# # Hint: Look up the numpy array_split function.                                #
# ################################################################################
# # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# pass

# # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# # A dictionary holding the accuracies for different values of k that we find
# # when running cross-validation. After running cross-validation,
# # k_to_accuracies[k] should be a list of length num_folds giving the different
# # accuracy values that we found when using that value of k.
# k_to_accuracies = {}


# ################################################################################
# # TODO:                                                                        #
# # Perform k-fold cross validation to find the best value of k. For each        #
# # possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# # where in each case you use all but one of the folds as training data and the #
# # last fold as a validation set. Store the accuracies for all fold and all     #
# # values of k in the k_to_accuracies dictionary.                               #
# ################################################################################
# # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# pass

# # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# # Print out the computed accuracies
# for k in sorted(k_to_accuracies):
#     for accuracy in k_to_accuracies[k]:
#         print('k = %d, accuracy = %f' % (k, accuracy))


# # In[ ]:


# # plot the raw observations
# for k in k_choices:
#     accuracies = k_to_accuracies[k]
#     plt.scatter([k] * len(accuracies), accuracies)

# # plot the trend line with error bars that correspond to standard deviation
# accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
# accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
# plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
# plt.title('Cross-validation on k')
# plt.xlabel('k')
# plt.ylabel('Cross-validation accuracy')
# plt.show()


# # In[ ]:


# # Based on the cross-validation results above, choose the best value for k,   
# # retrain the classifier using all the training data, and test it on the test
# # data. You should be able to get above 28% accuracy on the test data.
# best_k = 1

# classifier = KNearestNeighbor()
# classifier.train(X_train, y_train)
# y_test_pred = classifier.predict(X_test, k=best_k)

# # Compute and display the accuracy
# num_correct = np.sum(y_test_pred == y_test)
# accuracy = float(num_correct) / num_test
# print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))


# # **Inline Question 3**
# # 
# # Which of the following statements about $k$-Nearest Neighbor ($k$-NN) are true in a classification setting, and for all $k$? Select all that apply.
# # 1. The decision boundary of the k-NN classifier is linear.
# # 2. The training error of a 1-NN will always be lower than that of 5-NN.
# # 3. The test error of a 1-NN will always be lower than that of a 5-NN.
# # 4. The time needed to classify a test example with the k-NN classifier grows with the size of the training set.
# # 5. None of the above.
# # 
# # $\color{blue}{\textit Your Answer:}$
# # 
# # 
# # $\color{blue}{\textit Your Explanation:}$
# # 
# # 
