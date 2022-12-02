# %%
"""
## Assignment 1 
"""

# %%
"""
#### Rating Prediction
"""

# %%
import numpy
import urllib
import scipy.optimize
import random
import gzip
import csv
from collections import defaultdict
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# %%
# input file using gzip
path = "train_Interactions.csv.gz"
path = "/Users/gakiara/Desktop/258R/assignment1/train_Interactions.csv.gz"
f = gzip.open(path, "rt", encoding="utf8")
reader = csv.reader(f, delimiter=",")

# %%
# reading the file to build datasets
datasets = []
first = True
for line in reader:
    if first:
        header = line
        first = False
    else:
        d = dict(zip(header, line))
        # convert strings to integers for some fields
        d["rating"] = int(d["rating"])
        datasets.append(d)

# %%
# split the training data
data_train = datasets[:190000]
data_valid = datasets[190000:]

# %%
# computer the global mean
globalAverage = sum([d["rating"] for d in data_train]) / len(data_train)
globalAverage
# calculate initial value of alpha,beta_user and beta_book
alpha = globalAverage
userRatings = defaultdict(list)
bookRatings = defaultdict(list)

UserPerBook = defaultdict(list)
BookPerUser = defaultdict(list)
Pair_Rating = defaultdict(int)

for l in data_train:
    user, book = l["userID"], l["bookID"]
    UserPerBook[book].append(user)
    BookPerUser[user].append(book)
    userRatings[user].append(l["rating"])
    bookRatings[book].append(l["rating"])
    Pair_Rating[(user,book)] = int(l["rating"])


# beta_user,beta_book,beta_pair
userBias = defaultdict(float)
for u in userRatings:
    userBias[u] = globalAverage - (sum(userRatings[u]) / len(userRatings[u]))

bookBias = defaultdict(float)
for b in bookRatings:
    bookBias[b] = globalAverage - (sum(bookRatings[b]) / len(bookRatings[b]))


# %%
print(len(userBias))
print(len(bookBias))

# %%
# define function for optimization
def differenciate(globalAverage, userBias, bookBias, lamb1,lamb2, epsilon):
    end = False
    globalAverage_last = 0
    userBias_last = userBias
    bookBias_last = bookBias
    MSE_last = 0
    cost_last = 0
    dataset = data_train
    iters = 0
    while not end:
        # update alpha
        globalAverage = 0
        for i in dataset:
            user, book = i["userID"], i["bookID"]
            globalAverage += i["rating"] - userBias_last[user] - bookBias_last[book]

        globalAverage = globalAverage / len(dataset)

        # update beta_user
        num_book = defaultdict(int)
        for u in userBias:
            userBias[u] = 0
        for i in dataset:
            user, book = i["userID"], i["bookID"]
            num_book[user] += 1
            userBias[user] += i["rating"] - globalAverage - bookBias_last[book]
        for u in userBias:
            userBias[u] = userBias[u] / (lamb1 + num_book[u])

        # update beta_book
        num_user = defaultdict(int)
        for b in bookBias:
            bookBias[b] = 0
        for i in dataset:
            user, book = i["userID"], i["bookID"]
            num_user[book] += 1
            bookBias[book] += i["rating"] - globalAverage - userBias[user]
        for b in bookBias:
            bookBias[b] = bookBias[b] / (lamb2 + num_user[b])

        predictions = []
        
        for d in dataset:
            user = d["userID"]
            book = d["bookID"]
            if user in userBias and book in bookBias:
                result = globalAverage + userBias[user] + bookBias[book]
            else:
                result = globalAverage
            predictions.append(result)
        
        labels = [l["rating"] for l in dataset]
        differences = [(x - y) ** 2 for x, y in zip(predictions, labels)]
        MSE = sum(differences) / len(differences)
        
        cost = MSE
        for u in userBias:
            cost += lamb1 * (userBias[u]**2)
        for i in bookBias:
            cost += lamb2 * (bookBias[i]**2)
        iters+=1;
        
        if (
            abs(MSE - MSE_last) < epsilon
            and abs(cost - cost_last) < epsilon
        ):
            end = True
            print(iters)
        else:
            globalAverage_last = globalAverage
            userBias_last = userBias
            bookBias_last = bookBias
            MSE_last = MSE
            cost_last = cost

    return globalAverage, userBias, bookBias

# %%
def getDirty0(Pair_Rating, BookPerUser, UserPerBook, dirty_feature):
    dirty_u = []
    dirty_i = []
    [dirty_limit, rating_std, rating_bound] = dirty_feature
    for u in BookPerUser.keys():
        if dirty_limit < len(BookPerUser[u]):
            rating_list = []
            for i in BookPerUser[u]:
                rating_list.append(Pair_Rating[(u,i)])
            if numpy.std(rating_list) < rating_std:
                if numpy.mean(rating_list) > rating_bound[1] or numpy.mean(rating_list) < rating_bound[0]:
                    dirty_u.append(u)
    for i in UserPerBook.keys():
        if dirty_limit < len(UserPerBook[i]):
            rating_set = set()
            for u in UserPerBook[i]:
                rating_set.add(Pair_Rating[(u,i)])
            if len(rating_set) == 1:
                dirty_i.append(i)
    return [dirty_u, dirty_i]

# %%
def getDirty(Pair_Rating, BookPerUser, UserPerBook, dirty_feature):
    dirty_u = []
    dirty_i = []
    #[dirty_limit, rating_std, rating_bound] = dirty_feature
    [dirty_limit, rating_std] = dirty_feature
    for u in BookPerUser.keys():
        # data too sparse
        if len(BookPerUser[u]) < dirty_limit[0]:
            dirty_u.append(u)
        else:
            rating_list = []
            for i in BookPerUser[u]:
                rating_list.append(Pair_Rating[(u,i)])
            #data is not stable
            if numpy.std(rating_list) > rating_std:
                dirty_u.append(u)

    for i in UserPerBook.keys():
        if len(UserPerBook[i]) < dirty_limit[1]:
            dirty_i.append(i)

    return [dirty_u, dirty_i]

# %%
def intial_clean0(dataset, dirtyfeature):
    dataset_clean = []
    [dirty_u, dirty_i] = getDirty0(Pair_Rating, BookPerUser, UserPerBook, dirtyfeature)
    for d in dataset:
        user, book = d["userID"], d["bookID"]
        if user not in dirty_u and book not in dirty_i:
            dataset_clean.append(d)

    # computer the global mean
    globalAverage = sum([d["rating"] for d in dataset_clean]) / len(dataset_clean)
    # calculate initial value of alpha,beta_user and beta_book
    alpha = globalAverage
    userRatings = defaultdict(list)
    bookRatings = defaultdict(list)

    for l in dataset_clean:
        user, book = l["userID"], l["bookID"]
        userRatings[user].append(l["rating"])
        bookRatings[book].append(l["rating"])


    # beta_user,beta_book,beta_pair
    userBias = defaultdict(float)
    for u in userRatings:
        userBias[u] = globalAverage - (sum(userRatings[u]) / len(userRatings[u]))

    bookBias = defaultdict(float)
    for b in bookRatings:
        bookBias[b] = globalAverage - (sum(bookRatings[b]) / len(bookRatings[b]))

    return dataset_clean,globalAverage, userBias, bookBias

# %%
def intial_clean(dataset, dirtyfeature):
    dataset_clean = []
    [dirty_u, dirty_i] = getDirty(Pair_Rating, BookPerUser, UserPerBook, dirtyfeature)
    for d in dataset:
        user, book = d["userID"], d["bookID"]
        if user not in dirty_u and book not in dirty_i:
            dataset_clean.append(d)

    # computer the global mean
    globalAverage = sum([d["rating"] for d in dataset_clean]) / len(dataset_clean)
    # calculate initial value of alpha,beta_user and beta_book
    alpha = globalAverage
    userRatings = defaultdict(list)
    bookRatings = defaultdict(list)

    for l in dataset_clean:
        user, book = l["userID"], l["bookID"]
        userRatings[user].append(l["rating"])
        bookRatings[book].append(l["rating"])


    # beta_user,beta_book,beta_pair
    userBias = defaultdict(float)
    for u in userRatings:
        userBias[u] = globalAverage - (sum(userRatings[u]) / len(userRatings[u]))

    bookBias = defaultdict(float)
    for b in bookRatings:
        bookBias[b] = globalAverage - (sum(bookRatings[b]) / len(bookRatings[b]))

    return dataset_clean,globalAverage, userBias, bookBias

# %%
# define function for optimization
def differenciate2(globalAverage, userBias, bookBias, lamb1,lamb2, epsilon, dataset):

    end = False
    globalAverage_last = 0
    userBias_last = userBias
    bookBias_last = bookBias
    MSE_last = 0
    cost_last = 0

    iters = 0
    while not end:
        # update alpha
        globalAverage = 0
        for i in dataset:
            user, book = i["userID"], i["bookID"]
            globalAverage += i["rating"] - userBias_last[user] - bookBias_last[book]

        globalAverage = globalAverage / len(dataset)

        # update beta_user
        num_book = defaultdict(int)
        for u in userBias:
            userBias[u] = 0
        for i in dataset:
            user, book = i["userID"], i["bookID"]
            num_book[user] += 1
            userBias[user] += i["rating"] - globalAverage - bookBias_last[book]
        for u in userBias:
            userBias[u] = userBias[u] / (lamb1 + num_book[u])

        # update beta_book
        num_user = defaultdict(int)
        for b in bookBias:
            bookBias[b] = 0
        for i in dataset:
            user, book = i["userID"], i["bookID"]
            num_user[book] += 1
            bookBias[book] += i["rating"] - globalAverage - userBias[user]
        for b in bookBias:
            bookBias[b] = bookBias[b] / (lamb2 + num_user[b])

        predictions = []

        for d in dataset:
            user = d["userID"]
            book = d["bookID"]
            if user in userBias and book in bookBias:
                result = globalAverage + userBias[user] + bookBias[book]
            else:
                result = globalAverage
            predictions.append(result)

        labels = [l["rating"] for l in dataset]
        differences = [(x - y) ** 2 for x, y in zip(predictions, labels)]
        MSE = sum(differences) / len(differences)

        cost = MSE
        for u in userBias:
            cost += lamb1 * (userBias[u]**2)
        for i in bookBias:
            cost += lamb2 * (bookBias[i]**2)
        iters+=1;

        if (
            abs(MSE - MSE_last) < epsilon
            and abs(cost - cost_last) < epsilon
        ):
            end = True
            print(iters)
        else:
            globalAverage_last = globalAverage
            userBias_last = userBias
            bookBias_last = bookBias
            MSE_last = MSE
            cost_last = cost

    return globalAverage, userBias, bookBias

# %%
# prediction result
def prediction(user, book):
    if user in userBias and book in bookBias:
        result = globalAverage_new + userBias_new[user] + bookBias_new[book]
    else:
        result = globalAverage_new
    if result < 0.75: result = 0
    if result > 0.75 and result < 1.25:
        if  result > 0.9 and result < 1.1:
            result = 1
        else:
            result = 0
    if result > 1.9 and result < 2.1: result = 2
    if result > 4.9: result = 5

    return result

# %%
# prediction result
def prediction2(user, book):
    if user in userBias and book in bookBias:
        result = globalAverage_new + userBias_new[user] + bookBias_new[book]
    elif user in userBias:
        result = globalAverage_new + userBias_new[user]
    elif book in bookBias:
        result = globalAverage_new + bookBias_new[book]
    else:
        result = globalAverage_new
    if result < 1: result = 0
    #if result > 5: result = 5
    return result

# %%
print(random.uniform(0,1))

# %%
# define MSE function
def MSE(dataset):
    predictions = [prediction(d["userID"], d["bookID"]) for d in dataset]
    labels = [d["rating"] for d in dataset]
    differences = [(x - y) ** 2 for x, y in zip(predictions, labels)]
    mse = sum(differences) / len(differences)
    return mse

# %%
# define MSE function
def MSE2(dataset):
    predictions = [prediction2(d["userID"], d["bookID"]) for d in dataset]
    labels = [d["rating"] for d in dataset]
    differences = [(x - y) ** 2 for x, y in zip(predictions, labels)]
    mse = sum(differences) / len(differences)
    return mse

# %%
m = MSE(data_valid)
print(m)

# %%
globalAverage_new, userBias_new, bookBias_new = differenciate(globalAverage, userBias, bookBias, 3.87,14.76, epsilon=1e-7)

# %%
print(globalAverage_new)

# %%
m = MSE(data_valid)
print(m)

# %%
dirty_limits = [10,12,14,16,18,20]
sds = [0.05,0.08,0.1,0.2]
dirty_bound = [[1.5,4.5],[2,4],[2.6, 4.4],[2.5,3.5],[3,3]]
dirty_mse = defaultdict(list)
mse = []
for dl in dirty_limits:
    for s in sds:
        for db in dirty_bound:
             dirty_feature = [dl, s ,db]
             dataset_clean,globalAverage,userBias,userBias = intial_clean0(data_train,dirty_feature)
             globalAverage_new, userBias_new, bookBias_new = differenciate2(globalAverage, userBias, bookBias,3.87,14.76,1e-06,dataset_clean)
             m = MSE(data_valid)
             print(m,dl,s,db)
             print(len(dataset_clean))
             mse.append(m)
             dirty_mse[m] = dirty_feature

# %%
best_mse = min(mse)
best_df = dirty_mse[best_mse]
print("Optimal dirty feature:", best_df, "; MSE:", best_mse)

# %%
dirty_limits = [[2,2],[2,3],[2,4],[2,5]]
sds = [2.0,2.2,2.4,2.6,2.8,3,3.2]
#dirty_bound = [[2.6,4.4],[2.8,4.2],[3,4],[3.5,3.5]]
dirty_mse = defaultdict(list)
mse = []
for dl in dirty_limits:
    for s in sds:
        #for db in dirty_bound:
             dirty_feature = [dl, s ]
             dataset_clean,globalAverage,userBias,userBias = intial_clean(data_train,dirty_feature)
             globalAverage_new, userBias_new, bookBias_new = differenciate2(globalAverage, userBias, bookBias,3.87,14.67,1e-05,dataset_clean)
             m = MSE(data_valid)
             print(m,dl,s)
             print(len(dataset_clean))
             mse.append(m)
             dirty_mse[m] = dirty_feature

# %%
best_mse = min(mse)
best_df = dirty_mse[best_mse]
print("Optimal dirty feature:", best_df, "; MSE:", best_mse)

# %%
#Find the best lambda

# %%
lamb1 = [i for i in range(1,20)]
lamb2 = [i for i in range(1,20)]
mse = []
lambs_mse = defaultdict(list)
#when optimal the lamb1,epsilon should be bigger
for r1 in lamb1:
  for r2 in lamb2:
    globalAverage_new, userBias_new, bookBias_new = differenciate(
        globalAverage, userBias, bookBias, r1,r2, epsilon=1e-05
    )
    m = MSE(data_valid)
    lambs_mse[m]=[r1,r2]
    mse.append(m)

# %%
print(lambs_mse)

# %%
# Optimal λ
best_mse = min(mse)
best_lamda = lambs_mse[best_mse]
print("Optimal λ:", best_lamda, "; MSE:", best_mse)

# %%
#finetuning the lambda

# %%
# calculate initial value of alpha,beta_user and beta_book
# computer the global mean
# computer the global mean
globalAverage = sum([d["rating"] for d in data_train]) / len(data_train)
globalAverage

alpha = globalAverage

# beta_user,beta_book
userBias = defaultdict(float)
for u in userRatings:
    userBias[u] = globalAverage - (sum(userRatings[u]) / len(userRatings[u]))

bookBias = defaultdict(float)
for b in bookRatings:
    bookBias[b] = globalAverage - (sum(bookRatings[b]) / len(bookRatings[b]))

# %%
lamb1 = [i/100 for i in range(300,510)]
mse = []
lamb1_mse = defaultdict(list)
#when optimal the lamb1,epsilon should be bigger
for r1 in lamb1:
    globalAverage_new, userBias_new, bookBias_new = differenciate(
        globalAverage, userBias, bookBias, r1,15, epsilon=1e-05
    )
    m = MSE(data_valid)
    lamb1_mse[m]=[r1]
    mse.append(m)

# %%
print(lamb1_mse)

# %%
# Optimal λ
best_mse = min(mse)
best_lamda1 = lamb1_mse[best_mse]
print("Optimal λ1:", best_lamda1, "; MSE:", best_mse)

# %%
# calculate initial value of alpha,beta_user and beta_book
# computer the global mean
# computer the global mean
globalAverage = sum([d["rating"] for d in data_train]) / len(data_train)
globalAverage

alpha = globalAverage

# beta_user,beta_book
userBias = defaultdict(float)
for u in userRatings:
    userBias[u] = globalAverage - (sum(userRatings[u]) / len(userRatings[u]))

bookBias = defaultdict(float)
for b in bookRatings:
    bookBias[b] = globalAverage - (sum(bookRatings[b]) / len(bookRatings[b]))

# %%
lamb2 = [i/100 for i in range(1450,1510)]
mse = []
lamb2_mse = defaultdict(list)
#when optimal the lamb1,epsilon should be bigger
for r2 in lamb2:
    globalAverage_new, userBias_new, bookBias_new = differenciate(
        globalAverage, userBias, bookBias, 3.87, r2, epsilon=1e-05
    )
    m = MSE(data_valid)
    lamb2_mse[m]=[r2]
    mse.append(m)

# %%
print(lamb2_mse)

# %%
# Optimal λ
best_mse = min(mse)
best_lamda2 = lamb2_mse[best_mse]
print("Optimal λ2:", best_lamda2, "; MSE:", best_mse)

# %%
#finetuning the epsilon

# %%
mse = []
epsilon_mse = defaultdict()
epsilons = [10**(-i) for i in range(6,15)]
for epsilon in epsilons:
    globalAverage_new, userBias_new, bookBias_new = differenciate(
        globalAverage, userBias, bookBias, 3.87,14.76, epsilon
    )
    m = MSE(data_valid)
    epsilon_mse[m]=epsilon
    mse.append(m)

# %%
print(epsilon_mse)

# %%
# Optimal λ
best_mse = min(mse)
best_epsilon = epsilon_mse[best_mse]
print("Optimal epsilon:", best_epsilon, "; MSE:", best_mse)

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%
"""
#### USE ALL the DATA
"""

# %%
# computer the global mean
globalAverage = sum([d["rating"] for d in datasets]) / len(datasets)
globalAverage

# %%
# calculate initial value of alpha,beta_user and beta_book
alpha = globalAverage
userRatings = defaultdict(list)
bookRatings = defaultdict(list)

for l in datasets:
    user, book = l["userID"], l["bookID"]
    userRatings[user].append(l["rating"])
    bookRatings[book].append(l["rating"])

# beta_user,beta_book
userBias = defaultdict(float)
for u in userRatings:
    userBias[u] = globalAverage - (sum(userRatings[u]) / len(userRatings[u]))

bookBias = defaultdict(float)
for b in bookRatings:
    bookBias[b] = globalAverage - (sum(bookRatings[b]) / len(bookRatings[b]))

# %%
# define function for optimization
def differenciate(globalAverage, userBias, bookBias, lamb1,lamb2, epsilon):
    end = False
    globalAverage_last = 0
    userBias_last = userBias
    bookBias_last = bookBias
    MSE_last = 0
    cost_last = 0
    dataset = datasets
    while not end:
        # update alpha
        globalAverage = 0
        for i in dataset:
            user, book = i["userID"], i["bookID"]
            globalAverage += i["rating"] - userBias_last[user] - bookBias_last[book]

        globalAverage = globalAverage / len(dataset)

        # update beta_user
        num_book = defaultdict(int)
        for u in userBias:
            userBias[u] = 0
        for i in dataset:
            user, book = i["userID"], i["bookID"]
            num_book[user] += 1
            userBias[user] += i["rating"] - globalAverage - bookBias_last[book]
        for u in userBias:
            userBias[u] = userBias[u] / (lamb1 + num_book[u])

        # update beta_book
        num_user = defaultdict(int)
        for b in bookBias:
            bookBias[b] = 0
        for i in dataset:
            user, book = i["userID"], i["bookID"]
            num_user[book] += 1
            bookBias[book] += i["rating"] - globalAverage - userBias[user]
        for b in bookBias:
            bookBias[b] = bookBias[b] / (lamb2 + num_user[b])

        predictions = []

        for d in dataset:
            user = d["userID"]
            book = d["bookID"]
            if user in userBias and book in bookBias:
                result = globalAverage + userBias[user] + bookBias[book]
            else:
                result = globalAverage
            predictions.append(result)

        labels = [l["rating"] for l in dataset]
        differences = [(x - y) ** 2 for x, y in zip(predictions, labels)]
        MSE = sum(differences) / len(differences)

        cost = MSE
        for u in userBias:
            cost += lamb1 * (userBias[u]**2)
        for i in bookBias:
            cost += lamb2 * (bookBias[i]**2)


        if (
            abs(MSE - MSE_last) < epsilon
            and abs(cost - cost_last) < epsilon
        ):
            end = True
        else:
            globalAverage_last = globalAverage
            userBias_last = userBias
            bookBias_last = bookBias
            MSE_last = MSE
            cost_last = cost

    return globalAverage, userBias, bookBias

# %%

globalAverage_new, userBias_new, bookBias_new = differenciate(
    globalAverage, userBias, bookBias, 3.87,14.76, epsilon=1e-7)
'''
globalAverage_new, userBias_new, bookBias_new, pairBias_new = differenciate2(globalAverage, userBias, bookBias, pairBias, 3.853, 14.74, 600, epsilon=1e-07)
'''

# %%
predictions = open("/Users/gakiara/Desktop/258R/assignment1/predictions_Rating.csv", "w")
for l in open("/Users/gakiara/Desktop/258R/assignment1/pairs_Rating.csv"):
    if l.startswith("userID"):
        # header
        predictions.write(l)
        continue
    u, b = l.strip().split(",")
    p = prediction(u, b)
    predictions.write(u + "," + b + "," + str(p) + "\n")
predictions.close()
#1.4547551286

# %%
