import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from sklearn import svm

import processEmail as pe
import emailFeatures as ef

plt.ion()
np.set_printoptions(formatter={'float': '{: 0.6f}'.format})

# ===================== Part 1: Email Preprocessing =====================
# To use an SVM to classify emails into spam v. non-spam, you first need to
# convert each email into a vector of features. In this part, you will
# implement the preprocessing steps for each email. You should
# complete the code in processEmail.py to produce a word indices vector
# for a given email.

print('Preprocessing sample email (emailSample1.txt) ...')

file_contents = open('emailSample1.txt', 'r').read()
word_indices = pe.process_email(file_contents)

# Print stats
print('Word Indices: ')
print(word_indices)

input('Program paused. Press ENTER to continue')

# ===================== Part 2: Feature Extraction =====================
# Now, you will convert each email into a vector of features in R^n.
# You should complete the code in emailFeatures.py to produce a feature
# vector for a given mail

print('Extracting Features from sample email (emailSample1.txt) ... ')

# Extract features
features = ef.email_features(word_indices)

# Print stats
print('Length of feature vector: {}'.format(features.size))
print('Number of non-zero entries: {}'.format(np.flatnonzero(features).size))

input('Program paused. Press ENTER to continue')

# ===================== Part 3: Train Linear SVM for Spam Classification =====================
# In this section, you will train a linear classifier to determine if an
# email is Spam or Not-spam.

# Load the Spam Email dataset
# You will have X, y in your environment
data = scio.loadmat('spamTrain.mat')
X = data['X']
y = data['y'].flatten()

print('Training Linear SVM (Spam Classification)')
print('(this may take 1 to 2 minutes)')

c = 0.1
clf = svm.SVC(c, kernel='linear')
clf.fit(X, y)

p = clf.predict(X)

print('Training Accuracy: {}'.format(np.mean(p == y) * 100))

# ===================== Part 4: Test Spam Classification =====================
# After training the classifier, we can evaluate it on a test set. We have
# included a test set in spamTest.mat

# Load the test dataset
data = scio.loadmat('spamTest.mat')
Xtest = data['Xtest']
ytest = data['ytest'].flatten()

print('Evaluating the trained linear SVM on a test set ...')

p = clf.predict(Xtest)

print('Test Accuracy: {}'.format(np.mean(p == ytest) * 100))

input('Program paused. Press ENTER to continue')

# ===================== Part 5: Top Predictors of Spam =====================
# Since the model we are training is a linear SVM, we can inspect the w
# weights learned by the model to understand better how it is determining
# whether an email is spam or not. The following code finds the words with
# the highest weights in the classifier. Informally, the classifier
# 'thinks' that these words are the most likely indicators of spam.
#

vocab_list = pe.get_vocab_list()
indices = np.argsort(clf.coef_).flatten()[::-1]
print(indices)

for i in range(15):
    print('{} ({:0.6f})'.format(vocab_list[indices[i]], clf.coef_.flatten()[indices[i]]))

input('ex6_spam Finished. Press ENTER to exit')
