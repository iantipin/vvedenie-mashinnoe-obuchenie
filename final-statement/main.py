# coding=utf8

from __future__ import division

import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def get_unique_heroes( data ):
    unique_heroes = []

    for column_name in hero_columns:
        for v in data[column_name].value_counts().index:
            if v not in unique_heroes:
                unique_heroes.append( v )

    print ''
    print 'unique_heroes:', len( unique_heroes )

    return unique_heroes


def get_X_heroes( data ):
    unique_heroes = get_unique_heroes( data )
    number_heroes = len( unique_heroes )

    X_heroes = np.zeros( (data.shape[0], number_heroes) )

    for i, match_id in enumerate( data.index ):
        for p in xrange( 5 ):
            r_hero_id = data.ix[match_id, 'r%d_hero' % (p + 1)]
            X_heroes[i, unique_heroes.index( r_hero_id )] = 1

            d_hero_id = data.ix[match_id, 'd%d_hero' % (p + 1)]
            X_heroes[i, unique_heroes.index( d_hero_id )] = -1

    return X_heroes


def get_X_y( data_source, target_column, is_y = True, is_drop = False, is_fillna = False, is_scaler = False, is_X_heroes = False,
             drop_columns = [] ):
    if is_drop and len( drop_columns ):
        data = data_source.drop( columns = drop_columns )

        print 'drop'
    else:
        data = data_source

    if is_y:
        y = data[target_column]
        X = data.drop( columns = [target_column] )
    else:
        y = []
        X = data

    if is_fillna:
        X.fillna( 0, inplace = True )

        print 'fillna'

    if is_scaler:
        scaler = StandardScaler()

        X = scaler.fit_transform( X )

        print 'scaler'

    if is_X_heroes:
        X_heroes = get_X_heroes( data_source )

        X = np.hstack( (X, X_heroes) )

        print 'X_heroes'

    print ''

    return X, y


def print_incomplete_columns( data ):
    # train_df.isnull().sum()

    incomplete_columns = []

    count_rows = data.shape[0]

    for column_name in data.columns:
        count_values = data[column_name].count()

        if count_values < count_rows:
            incomplete_columns.append(
                [column_name, count_rows - count_values, '%.2f' % (count_rows / count_values * 100 - 100)] )

    df = pd.DataFrame( data = incomplete_columns, columns = ['Признак', 'Пропуски', 'Пропуски, %'] )

    print 'Признаки имеющие пропуски среди своих значений:'
    print ''
    print df.sort_values( by = ['Пропуски, %'], ascending = False )
    print ''


def print_scoring( model, X, y, scoring = 'roc_auc', label = '' ):
    cv_start_time = datetime.datetime.now()

    cv = KFold( n_splits = 5, shuffle = True, random_state = random_state )

    roc_auc = cross_val_score( model, X, y, cv = cv, scoring = scoring ).mean()

    print label, scoring, '%.2f' % roc_auc
    print datetime.datetime.now() - cv_start_time
    print ''


def print_best_params( model, param_grid, X, y, scoring = 'roc_auc' ):
    cv_start_time = datetime.datetime.now()

    cv = KFold( n_splits = 5, shuffle = True, random_state = random_state )

    gs = GridSearchCV( model, param_grid, cv = cv, scoring = scoring )

    gs.fit( X, y )

    print 'best_params_', gs.best_params_
    print datetime.datetime.now() - cv_start_time
    print ''


pd.set_option( 'display.width', 1024 )
random_state = 241
start_time = datetime.datetime.now()

print 'Time start:', start_time
print ''

overfitting_columns = ['duration', 'tower_status_radiant', 'tower_status_dire',
                       'barracks_status_radiant',
                       'barracks_status_dire']

hero_columns = ['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']

categorical_columns = [
    'lobby_type'
]
categorical_columns = np.hstack( (hero_columns, categorical_columns) )

useless_columns = ['start_time']

###################################
#
# Train
#
###################################

features = pd.read_csv( './data/features.csv', index_col = 'match_id' )

print_incomplete_columns( features )

target_column = 'radiant_win'

print 'Целевая переменная:', target_column
print ''
###################################
#
# Train, GradientBoostingClassifier
#
###################################
if False:
    drop_columns = np.hstack( (overfitting_columns, categorical_columns, useless_columns) )

    X_train, y_train = get_X_y( features, target_column = target_column, is_drop = True, is_fillna = True, is_scaler = False, is_X_heroes = False, drop_columns = drop_columns )

    model = GradientBoostingClassifier( n_estimators = 1000, max_depth = 1, random_state = random_state )

    if False:
        param_grid = { 'n_estimators': np.arange( 10, 40, 10 ) }

        print_best_params( model, param_grid, X_train, y_train, scoring = 'roc_auc' )  # 'n_estimators': 30

    if True:
        print_scoring( model, X_train, y_train, scoring = 'roc_auc', label = 'GradientBoostingClassifier' )

'''

Отчет:

1. Какие признаки имеют пропуски среди своих значений? Что могут означать пропуски в этих признаках (ответьте на этот вопрос для двух любых признаков)?

    Признаки имеющие пропуски среди своих значений:

    – first_blood_player2           # Признаки события "первая кровь" (first blood), второй игрок, причастный к событию
    – radiant_flying_courier_time   # Время приобретения предмета "flying_courier"
    – dire_flying_courier_time
    – first_blood_time              # Признаки события "первая кровь" (first blood), игровое время первой крови
    – first_blood_team
    – first_blood_player1
    – dire_bottle_time
    – radiant_bottle_time
    – radiant_first_ward_time
    – dire_first_ward_time
    – radiant_courier_time
    – dire_courier_time

В первые 5-ть минут игры в 75% матчей случается событие "первая кровь" и в 60% матчей покупают предмет "flying_courier".


2. Как называется столбец, содержащий целевую переменную?

    radiant_win


3. Как долго проводилась кросс-валидация для градиентного бустинга с 30 деревьями? Инструкцию по измерению времени можно найти ниже по тексту. Какое качество при этом получилось? Напомним, что в данном задании мы используем метрику качества AUC-ROC.
    
    2,7 GHz Intel Core i5
    8 ГБ 1867 MHz DDR3
    
    n_estimators = 30
    0:01:19
    roc_auc 0.69


4. Имеет ли смысл использовать больше 30 деревьев в градиентном бустинге? Что бы вы предложили делать, чтобы ускорить его обучение при увеличении количества деревьев?

Иеет смысл использовать больше 30 деревьев в градиентном бустинге:

    n_estimators = 1000
    0:42:49
    roc_auc 0.72

Чтобы ускорить обучение модели, можно уменьшить глубину деревьев в градиентом бустинге (max_depth):

    n_estimators = 1000, max_depth = 1
    0:09:52
    roc_auc 0.72

'''

###################################
#
# Train, LogisticRegression
#
###################################
if True:
    drop_columns = np.hstack( (overfitting_columns, categorical_columns) )

    X_train, y_train = get_X_y( features, target_column = target_column, is_drop = True, is_fillna = True, is_scaler = True, is_X_heroes = True, drop_columns = drop_columns )

    model = LogisticRegression( penalty = 'l2', C = 0.01, random_state = random_state )

    if True:
        print_scoring( model, X_train, y_train, scoring = 'roc_auc', label = 'LogisticRegression' )

    if False:
        param_grid = { 'C': np.arange( 0.01, 0.1, 0.01 ) }

        print_best_params( model, param_grid, X_train, y_train, scoring = 'roc_auc' )  # {'C': 0.01}

'''

Отчет:

1. Какое качество получилось у логистической регрессии над всеми исходными признаками? Как оно соотносится с качеством градиентного бустинга? Чем вы можете объяснить эту разницу? Быстрее ли работает логистическая регрессия по сравнению с градиентным бустингом?

    2,7 GHz Intel Core i5
    8 ГБ 1867 MHz DDR3
    
    0:00:14
    roc_auc 0.72
    
    Логистическая регрессия работает быстрее, при этом качество сопоставимо с качеством градиентного бустинга. Линейные методы работают гораздо быстрее композиций деревьев.
    

2. Как влияет на качество логистической регрессии удаление категориальных признаков (укажите новое значение метрики качества)? Чем вы можете объяснить это изменение?

    0:00:10
    roc_auc 0.72
    
    Удаление категориальных признаков не влияет на качество логистической регрессии.


3. Сколько различных идентификаторов героев существует в данной игре?

    108


4. Какое получилось качество при добавлении "мешка слов" по героям? Улучшилось ли оно по сравнению с предыдущим вариантом? Чем вы можете это объяснить?

    0:00:17
    roc_auc 0.75
    
    Качество улучшилось. Категориальные признаки трансформированы в матрицу признаков для каждого героя.


5. Какое минимальное и максимальное значение прогноза на тестовой выборке получилось у лучшего из алгоритмов?

    Min: 0.00713
    Max: 0.99307

'''

###################################
#
# Test, LogisticRegression
#
###################################
if True:
    features_test = pd.read_csv( './data/features_test.csv', index_col = 'match_id' )

    drop_columns = np.hstack( (categorical_columns) )

    X_test, y_empty = get_X_y( features_test, is_y = False, target_column = target_column, is_drop = True, is_fillna = True, is_scaler = True, is_X_heroes = True, drop_columns = drop_columns )

    model.fit( X_train, y_train )

    pred = model.predict_proba( X_test )

    print 'Min:', '%.5f' % min( pred[:, 1] )
    print 'Max:', '%.5f' % max( pred[:, 1] )
    print ''

###############################
#
# Train, CatBoostClassifier
#
###############################
if False:
    from catboost import CatBoostClassifier

    model = CatBoostClassifier( iterations = 1000, learning_rate = 0.5, depth = 3, verbose = None )

    if False:
        param_grid = { 'depth': np.arange( 1, 10, 1 ) }

        print_best_params( model, param_grid, X_train, y_train, scoring = 'roc_auc' )  # {'learning_rate': 0.60, 'iterations': 100}

    if False:
        print_scoring( model, X_train, y_train, scoring = 'roc_auc' )

print 'Time elapsed:', datetime.datetime.now() - start_time
