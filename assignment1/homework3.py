# %%
import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy
import string
import random
import string
from sklearn import linear_model

# %%
import warnings
warnings.filterwarnings("ignore")

# %%
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

# %%
def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)

# %%
def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r

# %%
answers = {}

# %%
# Some data structures that will be useful

# %%
allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)

# %%
len(allRatings)

# %%
ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ItemPerUser = defaultdict(set)
UserPerItem = defaultdict(set)
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)

books = set()
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))
    ItemPerUser[u].add(b)
    UserPerItem[b].add(u)
    books.add(b)

# %%
##################################################
# Rating prediction (CSE258 only)                #
##################################################

# %%
### Question 1

# %%
#Use a set to store all the book
books = set()
for u,b,r in allRatings:
    books.add(b)
books = list(books)
print(len(books),books[0])
Entry_Valid = []
#Pick a book randomly, maybe we do not need to one positive vs one negative
#we can make the number of nagative randomly
for u,b,r in ratingsValid:
    Entry_Valid.append((u,b,1))
    index = random.randint(0, len(books)-1)
    while books[index] in ItemPerUser[u]:
        index = random.randint(0, len(books)-1)
    if books[index] in ItemPerUser[u]:
        print("error")
    Entry_Valid.append((u,books[index],0))
print(len(Entry_Valid))

# %%
### Would-read baseline: just rank which books are popular and which are not, and return '1' if a book is among the top-ranked
bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("train_Interactions.csv.gz"):
  bookCount[book] += 1
  totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
  count += ic
  return1.add(i)
  if count > totalRead/2: break

# %%
#predict
#print(Entry_Valid)
acc = 0
for u,b,gt in Entry_Valid:
    if b in return1:
        pre = 1
    else:
        pre = 0
    if pre == gt:
        acc +=1
acc1 = acc/len(Entry_Valid)
print(acc1)

# %%
answers['Q1'] = acc1

# %%
assertFloat(answers['Q1'])

# %%
### Question 2

# %%
#set thresholds
thresholds = [i/100 for i in range(1,101)]
accs = []
print(thresholds)

# %%
for threshold in thresholds:
   return1 = set()
   count = 0
   for ic, i in mostPopular:
     count += ic
     return1.add(i)
     if count > totalRead*threshold: break
   acc = 0
   for u,b,gt in Entry_Valid:
     if b in return1:
          pre = 1
     else:
        pre = 0
     if pre == gt:
        acc +=1
     acc1 = acc/len(Entry_Valid)
   print(acc1)
   accs.append(acc1)

# %%
import matplotlib.pyplot as plt

plt.xlabel("Threshold Percentile")
plt.ylabel("Accuracy")
plt.plot(thresholds, accs)

# %%
best_threshold = thresholds[accs.index(max(accs))]
print("My best accurate %f" % max(accs))
print("The best threshold is %f " %best_threshold)

# %%
answers['Q2'] = [best_threshold, max(accs)]

# %%
assertFloat(answers['Q2'][0])
assertFloat(answers['Q2'][1])

# %%
### Question 3

# %%
def Jaccard(s1, s2):
    numerator = len(s1.intersection(s2))
    denominator = len(s1.union(s2))
    if(denominator == 0):
        return 0
    return numerator/denominator

# %%
def max_Jaccard(user, book):
    max_sim= 0
    for b_r in ItemPerUser[user]:
        sim = Jaccard(UserPerItem[book],UserPerItem[b_r])
        if sim > max_sim:
            max_sim = sim
    return max_sim
max_sim = {}
for u,b,_ in Entry_Valid:
    max_sim[(u,b)] = max_Jaccard(u, b)
print(max(max_sim.values()))

# %%
accs2 = []
thresholds2 = [i/10000 for i in range(1,101)]
#for i in range(1,11):
#    thresholds2.append(i/10)
print(thresholds2)

# %%
for threshold in thresholds2:
   acc = 0
   for u,b,gt in Entry_Valid:
     pre = 0
     if max_sim[(u,b)] >= threshold:
        pre = 1
     if pre == gt:
        acc +=1
     acc2 = acc/len(Entry_Valid)
   print(acc2)
   accs2.append(acc2)

# %%
plt.xlabel("Threshold Percentile")
plt.ylabel("Accuracy")
plt.plot(thresholds2, accs2)

# %%
max_acc2 = max(accs2)
best_threshold2 = thresholds2[accs2.index(max(accs2))]
print("My best accurate %f" % max_acc2)
print("The best threshold is %f " %best_threshold2)

# %%
answers['Q3'] = max_acc2

# %%
### Question 4

# %%
accs3 = []
best_threshold = 0.75
#best_thresholds = [i/100 for i in range(70,81)]
#best_threshold2 = 0.002930
best_threshold2 = 0.027780
best_threshold2s = [i/100000 for i in range(2001,4001)]

# %%
for best_threshold2 in best_threshold2s:
   return1 = set()
   count = 0
   for ic, i in mostPopular:
     count += ic
     return1.add(i)
     if count > totalRead*best_threshold: break
   acc = 0
   for u,b,gt in Entry_Valid:
     new_sim = max_sim[(u,b)]
     if b in return1 or new_sim >= best_threshold2:
         pre = 1
     else:
        pre = 0
     if pre == gt:
        acc +=1
     acc1 = acc/len(Entry_Valid)
   print(acc1)
   accs3.append(acc1)

# %%
plt.xlabel("Threshold Percentile")
plt.ylabel("Accuracy")
plt.plot(best_threshold2s, accs3)

# %%
max_acc3 = max(accs3)
best_threshold = best_threshold2s[accs3.index(max(accs3))]
print("My best accurate %f" % max_acc3)
print("The best threshold is %f " % best_threshold)

# %%
answers['Q4'] = max_acc3

# %%
### Question 5

# %%
predictions = open("predictions_Read.csv", 'w')
for l in open("pairs_Read.csv"):
    if l.startswith("userID"):
        # header
        predictions.write(l)
        continue
    u, b = l.strip().split(',')
    jac = max_Jaccard(u, b)
    if b in return1 or new_sim >= best_threshold:
       predictions.write(u + ',' + b + ",1\n")
    else:
       predictions.write(u + ',' + b + ",0\n")

predictions.close()

# %%
answers['Q5'] = 'I confirm that I have uploaded an assignment submission to gradescope'

# %%
### Question 9

# %%
print(allRatings[0])

# %%
ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ItemPerUser = defaultdict(set)
UserPerItem = defaultdict(set)
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)

users = set()
books = set()
Entry_Train = defaultdict(int)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))
    ItemPerUser[u].add(b)
    UserPerItem[b].add(u)
    Entry_Train[(u,b)] = int(r)
    books.add(b)
    users.add(u)

# %%
N = len(ratingsTrain)
print(N)

# %%
#iteration function
def cal_alpha(data_dict, beta_user, beta_item, N):
    sum = 0
    for  (user, item) , rating in data_dict.items():
        sum += rating - (beta_user[user] + beta_item[item])
    return sum/N

def cal_beta_user(data_dict, items_per_user, beta_item_dict, alpha, lamb):
    beta_user_dict = {}
    for user, items in items_per_user.items():
        sum = 0
        for item in items:
            rating = data_dict[(user, item)]
            beta_item = beta_item_dict[item]
            sum += rating - (alpha +beta_item)
        beta_user = sum / (lamb + len(items_per_user[user]))
        beta_user_dict[user] = beta_user
    return beta_user_dict

def cal_beta_item(data_dict, users_per_item, beta_user_dict, alpha, lamb):
    beta_item_dict = {}
    for item, users in users_per_item.items():
        sum = 0
        for user in users:
            rating = data_dict[(user, item)]
            beta_user = beta_user_dict[user]
            sum += rating - (alpha + beta_user)
        beta_item = sum/(lamb + len(users_per_item[item]))
        beta_item_dict[item] = beta_item
    return beta_item_dict

# %%
#initialize
beta_user = {u : 0 for u in users}
beta_item = {b : 0 for b in books}
lamb = 1

# %%
iterations = 1000
x=[]
y_alpha = []
y_bu = []
y_bi = []
for it in range(iterations):
    alpha = cal_alpha(Entry_Train,beta_user,beta_item,N)
    beta_user = cal_beta_user(Entry_Train,ItemPerUser,beta_item,alpha,lamb)
    beta_item = cal_beta_item(Entry_Train,UserPerItem,beta_user,alpha,lamb)
    x.append(it)
    y_alpha.append(alpha)
    y_bu.append(beta_user['u67805239'])
    y_bi.append(beta_item['b61372131'])

# %%
plt.xlabel("iterations")
plt.plot(x, y_alpha, label='Alpha')
plt.legend()

# %%
plt.xlabel("iterations")
plt.plot(x, y_bu)
plt.legend()

# %%
plt.xlabel("iterations")
plt.plot(x, y_bi)
plt.legend()

# %%
y = []
y_pred = []

# %%
def MSE(Y,Y_pred):
    mse = numpy.square(numpy.subtract(numpy.array(Y),numpy.array(Y_pred))).mean()
    return mse

# %%
for u,b,r in ratingsValid:
    beta_u = beta_user[u] if u in users else 0
    beta_i = beta_item[b] if b in books else 0
    y.append(r)
    pred = alpha + beta_u + beta_i
    y_pred.append(pred)
validMSE = MSE(y,y_pred)
print(validMSE)

# %%
answers['Q9'] = validMSE

# %%
assertFloat(answers['Q9'])

# %%
### 10

# %%
min_value = math.inf
min_id = 0
max_value = -math.inf
max_id = 0

for u in beta_user.keys():
    if min_value > beta_user[u]:
        min_value = beta_user[u]
        min_id = u
    if max_value < beta_user[u]:
        max_value = beta_user[u]
        max_id = u
maxUser = max_id
minUser = min_id
maxBeta = max_value
minBeta = min_value

# %%
answers['Q10'] = [maxUser, minUser, maxBeta, minBeta]
assert [type(x) for x in answers['Q10']] == [str, str, float, float]
print(answers['Q10'])

# %%
### Question 11

# %%
lambs = [10**i for i in range(-5,6)]
print(lambs)
mses = []
mses_lamb = defaultdict(float)

# %%
for lamb in lambs:
  iterations = 1000
  #initialize
  beta_user = {u : 0 for u in users}
  beta_item = {b : 0 for b in books}
  for it in range(iterations):
    alpha = cal_alpha(Entry_Train,beta_user,beta_item,N)
    beta_user = cal_beta_user(Entry_Train,ItemPerUser,beta_item,alpha,lamb)
    beta_item = cal_beta_item(Entry_Train,UserPerItem,beta_user,alpha,lamb)
  y = []
  y_pred = []
  for u,b,r in ratingsValid:
    beta_u = beta_user[u] if u in users else 0
    beta_i = beta_item[b] if b in books else 0
    y.append(r)
    pred = alpha + beta_u + beta_i
    y_pred.append(pred)
  validMSE = MSE(y,y_pred)
  mses.append(validMSE)
  mses_lamb[validMSE]=lamb

# %%
mses.sort()
print(mses)
Best_lamb = mses_lamb[mses[0]]
print(Best_lamb)
lamb = Best_lamb
validMSE = mses[0]

# %%
answers['Q11'] = (lamb, validMSE)

# %%
assertFloat(answers['Q11'][0])
assertFloat(answers['Q11'][1])

# %%
iterations = 1000
#initialize
beta_user = {u : 0 for u in users}
beta_item = {b : 0 for b in books}
for it in range(iterations):
    alpha = cal_alpha(Entry_Train,beta_user,beta_item,N)
    beta_user = cal_beta_user(Entry_Train,ItemPerUser,beta_item,alpha,lamb)
    beta_item = cal_beta_item(Entry_Train,UserPerItem,beta_user,alpha,lamb)

# %%
predictions = open("predictions_Rating.csv", 'w')
for l in open("pairs_Rating.csv"):
    if l.startswith("userID"): # header
        predictions.write(l)
        continue
    u,b = l.strip().split(',') # Read the user and item from the "pairs" file and write out your prediction
    # (etc.)
    beta_u = beta_user[u] if u in users else 0
    beta_i = beta_item[b] if b in books else 0
    pred = alpha + beta_u + beta_i
    predictions.write(u+","+b+","+str(pred)+"\n")
predictions.close()

# %%
# Copied from baseline code
bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalRead/2: break

# %%
f = open("answers_hw3.txt", 'w')
f.write(str(answers) + '\n')
f.close()

# %%
