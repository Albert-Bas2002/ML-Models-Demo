import csv
import random
from collections import Counter
import re
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score



train_x=[]
train_x_small=[]

train_y=[]
with open('RuReviews.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        row = next(reader)
        delimiter = "\t"
        row_last=row[-1]
        row=row[:-1]
        left_string, right_string = row_last.split(delimiter, 1)
        train_x.append(left_string.lower())
        if len(right_string) < 10:#Проверка длины
            train_y.append(right_string.lower())
        if row:
            for r in row:
                train_x.append(r.lower())
                train_y.append(right_string.lower())

print(train_x[0])
zipped = list(zip(train_x, train_y))
random.shuffle(zipped)
train_x, train_y = zip(*zipped)

test_size = 20000
test_x, train_x = train_x[:test_size], train_x[test_size:]#разделяем  данные на тестовые и для обуччения
test_y, train_y = train_y[:test_size], train_y[test_size:]

for element in train_x:
    middle = len(element) // 2 #создаем укороченный вариант предложений
    element_small = element[:middle]
    train_x_small.append(element_small)

def ZeroRule(test_x,train_x,train_y,test_y):
    top_y=Counter(train_y).most_common(1)[0][0]#находим самое часто встречающееся значения и присваиваем его каждому
    num=0
    for i in range((len(test_x))):
        if test_y[i]==top_y: #смотрим на точность
            num=num+1
    return str(num/ len(test_y)*100)+'%'

def OneRule(test_x,train_x,train_y,test_y):
    n=100
    classes = list(Counter(train_y).keys())
    top_y = Counter(train_y).most_common(1)[0][0]#находим наиболее частый класс
    dict_words={}
    predict=[]
    for one_class in classes:# цикл в котором создается словарь, где для каждого класса будут наиболее частые слова
        tmp_words=[]
        for idx,element in enumerate(train_y):
            if element==one_class:
                tmp_words.extend(sent_to_word(train_x[idx]))
        dict_words[one_class]=[key for key, value in Counter(tmp_words).most_common(n)]

    for idx,tst_x in enumerate(test_x): #функция, где мы находим сколько частых слов мы встретили в тестовой строке
        tmp_list={}
        for keys,lists in dict_words.items():
            tmp_list[keys]=len(np.intersect1d(sent_to_word(tst_x), lists))
            #если частых слов равное количество или не нашлось и одного слова, то используем самый частый класс(top_y)
        if Counter(tmp_list).most_common(1)[0][1]==Counter(tmp_list).most_common(2)[1][1] or Counter(tmp_list).most_common(1)[0][1]==0:
            predict.append(top_y)
        else:predict.append(Counter(tmp_list).most_common(1)[0][0])#у кого больше всего, тот и класс

    num = sum(1 for idx, pr in enumerate(predict) if pr == test_y[idx])
    return str(num/ len(test_y)*100)+'%'

def naiveB(test_x,train_x,train_y,test_y):

    # Создаем векторайзер и преобразуем предложения в матрицу признаков
    # vectorizer = CountVectorizer()
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(train_x)
    # Обучаем наивный байесовский классификатор
    clf = MultinomialNB()
    clf.fit(features, train_y)
    # Преобразуем новые предложения в матрицу признаков и классифицируем их
    new_features = vectorizer.transform(test_x)
    predict = clf.predict(new_features)
    # Выводим результаты классификации
    num = sum(1 for idx, pr in enumerate(predict) if pr == test_y[idx])
    precision = precision_score(test_y, predict, average=None)
    recall = recall_score(test_y, predict, average=None)
    print()
    print ('            positive     negativ     neautral')
    print('Precision:',precision)
    print('Recall:', recall)
    return str(num / len(test_y) * 100) + '%'

def SVC_alg(test_x,train_x,train_y,test_y):

    # Создаем векторайзер и преобразуем предложения в матрицу признаков
    #vectorizer = CountVectorizer()
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(train_x)
    # Обучаем наивный байесовский классификатор
    clf = SVC(kernel='poly', degree=3, coef0=1, C=2)
    #clf = SVC(kernel='rbf', gamma='scale')    # Обучение модели на тренировочных данных
    clf.fit(features, train_y)
    distances = clf.decision_function(features)
    threshold = np.percentile(distances, q=0.2)  # порог
    outliers = np.where(distances <= threshold)[0]
    print()
    print('________________Выбросы_____________________')
    #for outlier in outliers:print(train_x[outlier])
    print(len(outliers))#количество выбросов в данных, плюс svc что позволяет узнавать такие выбросы
    print('________________Выбросы_____________________')
    print()
    support_vectors=clf.support_vectors_
    print('Опорные вектора по признакам',support_vectors.shape)
    # Предсказание меток классов для тестовых данных
    # Преобразуем новые предложения в матрицу признаков и классифицируем их
    new_features = vectorizer.transform(test_x)
    predict = clf.predict(new_features)
    # Выводим результаты классификации
    num = sum(1 for idx, pr in enumerate(predict) if pr == test_y[idx])
    precision = precision_score(test_y, predict, average=None)
    recall = recall_score(test_y, predict, average=None)
    print()
    print('            positive     negativ     neautral')
    print('Precision:', precision)
    print('Recall:', recall)
    print()
    return str(num / len(test_y) * 100) + '%'

def KNN_alg(test_x,train_x,train_y,test_y):

    # Создаем векторайзер и преобразуем предложения в матрицу признаков
    # vectorizer = CountVectorizer()
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(train_x)
    # Обучаем наивный байесовский классификатор
    knn = KNeighborsClassifier(n_neighbors=int(np.sqrt(len(test_x))))
    knn.fit(features, train_y)
    # Преобразуем новые предложения в матрицу признаков и классифицируем их
    new_features = vectorizer.transform(test_x)
    predict = knn.predict(new_features)
    # Выводим результаты классификации
    num = sum(1 for idx, pr in enumerate(predict) if pr == test_y[idx])
    precision = precision_score(test_y, predict, average=None)
    recall = recall_score(test_y, predict, average=None)
    print()
    print('            positive     negativ     neautral')
    print('Precision:', precision)
    print('Recall:', recall)
    print()
    return str(num / len(test_y) * 100) + '%'

def DTree(test_x,train_x,train_y,test_y):

    # Создаем векторайзер и преобразуем предложения в матрицу признаков
    # vectorizer = CountVectorizer()
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(train_x)

    # Обучаем наивный байесовский классификатор
    dtr = DecisionTreeClassifier()
    dtr.fit(features, train_y)
    # Преобразуем новые предложения в матрицу признаков и классифицируем их
    new_features = vectorizer.transform(test_x)
    predict = dtr.predict(new_features)
    # Выводим результаты классификации
    num = sum(1 for idx, pr in enumerate(predict) if pr == test_y[idx])
    precision = precision_score(test_y, predict, average=None)
    recall = recall_score(test_y, predict, average=None)
    print()
    print('            positive     negativ     neautral')
    print('Precision:', precision)
    print('Recall:', recall)
    print()
    return str(num / len(test_y) * 100) + '%'


def sent_to_word(sentence):
    return re.findall('([А-яA-z]+)', sentence.lower())

print('ZeroRule',ZeroRule(test_x=test_x,train_x=train_x,train_y=train_y,test_y=test_y))
print('OneRule',OneRule(test_x=test_x,train_x=train_x,train_y=train_y,test_y=test_y))
print('naiveB',naiveB(test_x=test_x,train_x=train_x,train_y=train_y,test_y=test_y))
print('DTree',DTree(test_x=test_x,train_x=train_x,train_y=train_y,test_y=test_y))
print('KNN',KNN_alg(test_x=test_x,train_x=train_x,train_y=train_y,test_y=test_y))
print()
print('ZeroRule for 10000',ZeroRule(test_x=test_x,train_x=train_x[:10000],train_y=train_y[:10000],test_y=test_y))
print('OneRule for 10000',OneRule(test_x=test_x,train_x=train_x[:10000],train_y=train_y[:10000],test_y=test_y))
print('naiveB for 10000',naiveB(test_x=test_x,train_x=train_x[:10000],train_y=train_y[:10000],test_y=test_y))
print('DTree for 10000',DTree(test_x=test_x,train_x=train_x[:10000],train_y=train_y[:10000],test_y=test_y))
print('KNN for 10000',KNN_alg(test_x=test_x,train_x=train_x[:10000],train_y=train_y[:10000],test_y=test_y))
print()
print('ZeroRule for small sentence',ZeroRule(test_x=test_x,train_x=train_x_small,train_y=train_y,test_y=test_y))
print('OneRule for small sentence',OneRule(test_x=test_x,train_x=train_x_small,train_y=train_y,test_y=test_y))
print('naiveB for small sentence',naiveB(test_x=test_x,train_x=train_x_small,train_y=train_y,test_y=test_y))
print('DTree for small sentence',DTree(test_x=test_x,train_x=train_x_small,train_y=train_y,test_y=test_y))
print('KNN for small sentence',KNN_alg(test_x=test_x,train_x=train_x_small,train_y=train_y,test_y=test_y))
print()
print('SVC for 10000',SVC_alg(test_x=test_x,train_x=train_x[:10000],train_y=train_y[:10000],test_y=test_y))
print('SVC for small sentence',SVC_alg(test_x=test_x,train_x=train_x_small,train_y=train_y,test_y=test_y))
print('SVC',SVC_alg(test_x=test_x,train_x=train_x,train_y=train_y,test_y=test_y))