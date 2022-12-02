# %%
"""
## Assignment 1 
"""

# %%
"""
#### Read prediction
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
f = gzip.open(path,"rt",encoding="utf8")
reader = csv.reader(f,delimiter = ",")

# %%
# reading the file to build dataset
dataset = []
first = True
for line in reader:
    if first:
        header = line
        first = False
    else:
        d = dict(zip(header,line))
        # convert strings to integers for some fields
        d["rating"] = int(d["rating"])
        dataset.append(d)

# %%
# split the training data
data_train = dataset[:190000]
data_valid = dataset[190000:]

# %%
print(len(data_train))

# %%
# find all user ids and book ids, and pair them
allUserID=[]
allBookID=[]
UsersReadBooks = {}

for i in dataset:
    allUserID.append(i["userID"])
    allBookID.append(i["bookID"])
    if UsersReadBooks.get(i["userID"]):
        UsersReadBooks[i["userID"]].append(i["bookID"])
    else:
        UsersReadBooks[i["userID"]] = [i["bookID"]]

unique_users = list(set(allUserID))
unique_books = list(set(allBookID))

# %%
#validation data
Entry_Valid = []
positive_valid = []
for i in data_valid:
    positive_valid.append([i["userID"],i["bookID"]])
    u,b = i["userID"],i["bookID"]
    #positive sample
    Entry_Valid.append((u,b,1))
    index = random.randint(0, len(unique_books)-1)
    while unique_books[index] in UsersReadBooks[u]:
        index = random.randint(0, len(unique_books)-1)
    if unique_books[index] in UsersReadBooks[u]:
        print("error")
    Entry_Valid.append((u,unique_books[index],0))

print(len(Entry_Valid))


# %%
# negative validation data
negative_valid_dict = {}
for c in data_valid:
    bid = random.choice(unique_books)
    uid = c["userID"]
    while bid in UsersReadBooks[c["userID"]]:
        bid = random.choice(unique_books)
    if negative_valid_dict.get(uid):
        negative_valid_dict[uid].append(bid)
    else:
        negative_valid_dict[uid]= [bid]
        
# negative validation dataset to list
negative_valid = []

for i in negative_valid_dict.keys():
    if len(negative_valid_dict[i]) > 1:
        for ii in negative_valid_dict[i]:
            negative_valid.append([i,ii])
    else:
        negative_valid.append([i,negative_valid_dict[i][0]])

# %%
y_valid = [0 for i in range(len(negative_valid))] + [1 for i in range(len(positive_valid))]
X_valid = negative_valid + positive_valid 

# %%
"""
#### Book popularity
"""

# %%
# Baseline - using train dataset to get the most popular books data
bookCount = defaultdict(int)
userCount = defaultdict(int)
totalRead = 0
books = []

for c in dataset:
    user,book = c["userID"],c["bookID"]
    bookCount[book] += 1
    userCount[user] += 1
    totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

Average_book_Count = 0
for x in bookCount:
    Average_book_Count += bookCount[x]
Average_book_Count = Average_book_Count/len(bookCount)

book_popular = []
for i in range(int(len(mostPopular)*0.5)):
    book_popular.append(mostPopular[i])
print(Average_book_Count)

# %%
"""
#### Jaccard Similarity
"""

# %%
# pair users and books in dataset
TrainUserID = []
TrainBookID = []

UsersPerBook = defaultdict(set)
BooksPerUser = defaultdict(set)

for i in data_train:
    TrainUserID.append(i["userID"])
    TrainBookID.append(i["bookID"])
    UsersPerBook[i["bookID"]].add(i["userID"])
    BooksPerUser[i["userID"]].add(i["bookID"])

unique_users_train = list(set(TrainUserID))
unique_books_train = list(set(TrainBookID))

# %%
# Jaccard Predictor
def Jaccard(s1, s2):
    numerator = len(s1.intersection(s2))
    denominator = len(s1.union(s2))
    if(denominator == 0):
        return 0
    return numerator/denominator

def mostSimilarFast(user, book):
    similarities = []
    books = BooksPerUser[user]

    for b in books:
        if b == book:
            continue
        users = UsersPerBook[b]
        sim = Jaccard(users, UsersPerBook[book])
        similarities.append(sim)

    if len(similarities) > 0 :
         mean = sum(similarities)/len(similarities)
    else:
         mean = 0
   
    return mean

def mostSimilarFast_improve(user, book):
    similarities = []

    books = BooksPerUser[user]
    users = UsersPerBook[book]

    for b in books:
        if b == book: continue

        users = UsersPerBook[b]
        sim = Jaccard(users, UsersPerBook[book])
        similarities.append(sim)

    if len(similarities) > 0 :
         mean = sum(similarities)/len(similarities)
    else:
         #sometimes books are unseen
         for u in users:
            if u == user: continue
            books = BooksPerUser[u]
            sim = Jaccard(books, BooksPerUser[user])
            similarities.append(sim)
            if len(similarities) > 0 :
               mean = sum(similarities)/len(similarities)
            else:
               mean = 0
               print("error")

    return mean

def mostSimilarFast_improve2(user, book, alpha = 0.5):
    similarities1 = []
    similarities2 = []

    books = BooksPerUser[user]
    users = UsersPerBook[book]

    for b in books:
        if b == book: continue

        users = UsersPerBook[b]
        sim = Jaccard(users, UsersPerBook[book])
        similarities1.append(sim)

    if len(similarities1) > 0 :
           mean1 = sum(similarities1)/len(similarities1)
    else:
           mean1 = 0

    for u in users:
        if u == user: continue
        books = BooksPerUser[u]
        sim = Jaccard(books, BooksPerUser[user])
        similarities2.append(sim)

    if len(similarities2) > 0 :
           mean2 = sum(similarities2)/len(similarities2)
    else:
           mean2 = 0

    #there is a factor alpha that decides which part is more important in the similarity rating
    mean = alpha * mean1 + (1-alpha) * mean2

    return mean

# %%
"""
#### Prediction
"""

# %%
# test on validation dataset
# from homework3 we know bookcount is an important factor
user_book_sim = defaultdict(list)
#for x in X_valid:
for u,b,r in Entry_Valid:
    #s = mostSimilarFast(u,b) * bookCount[b]
    #s = mostSimilarFast_improve(u,b) * bookCount[b]
    u,b = d["userID"],d["bookID"]
    s = mostSimilarFast_improve2(u,b,0.97) * bookCount[b]
    comb = (u,b)
    user_book_sim[u].append ((s,comb))

for i in user_book_sim:
    user_book_sim[i].sort()

# %%
# fintuning
# In the first try, the best occurs when alphas in 0.9 - 1
alphas = [i/100 for i in range(90,101)]
accs = []
alpha_acc = defaultdict()
threshold = 0.5
for alpha in alphas:
    user_book_sim = defaultdict(list)
    for x in X_valid:
       u = x[0]
       b = x[1]
       #s = mostSimilarFast(u,b) * bookCount[b]
       #s = mostSimilarFast_improve(u,b) * bookCount[b]
       s = mostSimilarFast_improve2(u,b,alpha) * bookCount[b]
       comb = (u,b)
       user_book_sim[u].append ((s,comb))

    for i in user_book_sim:
         user_book_sim[i].sort()

    y_pred = []
    for x in X_valid:
       u = x[0]
       b = x[1]
       i = (u,b)
       book_sim_list = user_book_sim[u]
       p = 1
       for n in range(int(len(book_sim_list)*threshold)):
           if i in book_sim_list[n]:
               p = 0
       y_pred.append(p)
    acc = accuracy_score(y_pred, y_valid)
    accs.append(acc)
    alpha_acc[acc] = alpha

# %%
max_acc = max(accs)
best_alpha = alpha_acc[max_acc]
print("My best accurate %f" % max_acc)
print("The best alpha is %f " % best_alpha)

# %%
print(alpha_acc)

# %%
SimilarFast = defaultdict()
for u,b,r in Entry_Valid:
     SimilarFast[(u,b)] = mostSimilarFast_improve2(u,b,0.97)

user_book_sim = defaultdict(list)
user_book_sim1 = defaultdict(list)
user_book_sim2 = defaultdict(list)

for u,b,r in Entry_Valid:
    s = mostSimilarFast_improve2(u,b,0.97) * bookCount[b]
    s1 = bookCount[b]
    comb = (u,b)
    user_book_sim[u].append((s,comb))
    user_book_sim1[u].append(s1)

for u,b,r in Entry_Valid:
   #if(len(user_book_sim1[u])==2):
       books = BooksPerUser[u]
       J_max = 0
       max_book = b
       for book in books:
           if book == b: continue
           J_sim = Jaccard(UsersPerBook[book],UsersPerBook[b])
           if J_sim > J_max:
               J_max = J_sim
               max_book = book
       if max_book != b:
          s1 = bookCount[b]
          user_book_sim1[u].append(s1)
    #user_book_sim2[u].append(s1)

# %%
for i in user_book_sim:
    user_book_sim[i].sort()

for i in user_book_sim1:
    user_book_sim1[i].sort()

# %%
# user_book_sim(X_valid)

# %%
thresholds = [i/100 for i in range(1,101)]
thresholds = [0.583]

# %%
accs = []
threshold_acc = defaultdict()
error_count = defaultdict(int)
error_count2 = defaultdict(int)

for threshold in thresholds:

   y_pred = []
   y_valid = []
   #for x in X_valid:
   for u,b,r in Entry_Valid:
       i = (u,b)
       book_sim_list = user_book_sim[u]
       book_sim_list1 = user_book_sim1[u]
       p = 1
       for n in range(int(len(book_sim_list)*threshold)):
          if i in book_sim_list[n]:
               p = 0
       if(p==0 and int(r)==1):
            if len(book_sim_list) in error_count:
                error_count[len(book_sim_list)]+=1
            else:
                error_count[len(book_sim_list)]=1

       if(p==1 and int(r)==0):
            if len(book_sim_list) in error_count2:
                error_count2[len(book_sim_list)]+=1
            else:
                error_count2[len(book_sim_list)]=1



       y_pred.append(p)
       y_valid.append(int(r))

   acc = accuracy_score(y_pred, y_valid)
   accs.append(acc)
   threshold_acc[acc] = threshold

# %%
max_acc = max(accs)
best_threshold = threshold_acc[max_acc]
print("My best accurate %f" % max_acc)
print("The best threshold is %f " % best_threshold)

# %%
print(error_count)

# %%
print(error_count2)

# %%
max_acc = max(accs)
best_threshold = threshold_acc[max_acc]
print("My best accurate %f" % max_acc)
print("The best threshold is %f " % best_threshold)

# %%
"""
#### Use all the dataset
"""

# %%
# pair users and books in dataset
UsersPerBook = defaultdict(set)
BooksPerUser = defaultdict(set)

for i in dataset:
    UsersPerBook[i["bookID"]].add(i["userID"])
    BooksPerUser[i["userID"]].add(i["bookID"])

SimilarFast_total = defaultdict()
for u,b,r in dataset:
     SimilarFast_total[(u,b)] = mostSimilarFast_improve2(u,b,0.97)

# %%
"""
#### predict on the testing set
"""

# %%
X_test = []
for l in open("pairs_Read.csv"):
    if l.startswith("userID"):
        # header
#         predictions.write(l)
        continue
    u, b = l.strip().split(",")
    X_test.append([u,b])

# %%
y_pred = []
   #for x in X_valid:
total = 0
for x in X_test:
       u = x[0]
       b = x[1]
       i = (u,b)
       book_sim_list = user_book_sim[u]
       book_sim_list1 = user_book_sim1[u]

       f1 = 1
       f2 = 1

       if len(book_sim_list1) > 8:
           index = int(len(book_sim_list1)*0.599)
       elif len(book_sim_list1) == 8:
           index = 3
       elif len(book_sim_list1) == 6:
           index = 2
       elif len(book_sim_list1) == 4:
           index = 1
       else:
           index = 0

       item = bookCount[b]

       if item <= book_sim_list1[index]:
           f1 = 0

       if(f1 == 0 and item > 150):
           total += 1
           f1 = 1

       if f1 == 0:
           p = 0
       else:
           p = 1

       y_pred.append(p)

print(total)
print(len(y_pred))

# %%
predictions = open("predictions_Read.csv", "w")
n = -2
for l in open("pairs_Read.csv"):
    n += 1
    if l.startswith("userID"):
        # header
        predictions.write(l)
        continue
    u, b = l.strip().split(",")
    p = y_pred[n]
    predictions.write(u + "," + b + "," + str(p) + "\n")

predictions.close()


# %%
best_alpha = 0.97
best_threshold = 0.599
#best_threshold = 0.583
#best_threshold = 0.624  0.584  0.599-0.83
best_threshold2 = 0.4

# %%
user_book_sim = defaultdict(list)
user_book_sim1 = defaultdict(list)
for x in X_test:
    u = x[0]
    b = x[1]
    s = bookCount[b]
    s1 = mostSimilarFast_improve2(u,b,best_alpha) * bookCount[b]
    s2 = bookCount[b]
    comb = (u,b)
    user_book_sim[u].append((s1,comb))
    user_book_sim1[u].append(s2)

for i in user_book_sim:
    user_book_sim[i].sort()

for i in user_book_sim1:
    user_book_sim1[i].sort()

# %%
y_pred = []
total = 0
for x in X_test:
    u = x[0]
    b = x[1]
    i = (u,b)
    book_sim_list = user_book_sim[u]
    book_sim_list1 = user_book_sim1[u]
    p = 1
    for n in range(int(len(book_sim_list)*best_threshold)):
        if i in book_sim_list[n]:
            p = 0
    '''
    #when the book is so popular, predict to be 1
    for n in range(int(len(book_sim_list1)*best_threshold2)):
         if i in book_sim_list1[n]:
            p = 0
    '''
     # when the bookCount is too small
    if bookCount[b] < (sum(book_sim_list1)/len(book_sim_list1))*0.04:
              p = 0

    # when the bookCount is bigger than the mean
    if bookCount[b] > (sum(book_sim_list1)/len(book_sim_list1))*1.6:
              p = 1


    y_pred.append(p)
#0.599 - 0.8298
#0.599 0.04 1.9 -  0.8299
#0.599 0.04 1.77 - 0.83 - 0.0007(2/6-33.3%)
#0.599 0.04 1.6 - 0.8301  - (3/7 42.857%)

# %%
print(len(y_pred))

# %%
predictions = open("predictions_Read.csv", "w")
n = -2
for l in open("pairs_Read.csv"):
    n += 1
    if l.startswith("userID"):
        # header
        predictions.write(l)
        continue
    u, b = l.strip().split(",")
    p = y_pred[n]
    predictions.write(u + "," + b + "," + str(p) + "\n")

predictions.close()


# %%
#0.83