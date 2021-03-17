import sys
from sklearn import tree
from sklearn.model_selection import train_test_split
sys.path.append('data/')
from read_PLAID_data import read_processed_data

x, y = read_processed_data('type',
                           selected_label=[
                               'Fridge',
                               'Air Conditioner',
                           ],
                           direaction=1,
                           offset=30,
                           each_lenth=1)
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.5,
                                                    random_state=0)

for md in range(1, 10):
    for msl in [1, 5, 10, 15, 20, 25, 50, 100]:
        decision_tree = tree.DecisionTreeClassifier(max_depth=md,
                                                    min_samples_leaf=msl)
        decision_tree.fit(x_train, y_train)
        score = decision_tree.score(x_test, y_test)
        print('[max_depth:%02d,min_samples_leaf:%02d] score %.6f' %
              (md, msl, score))
        y_predict = decision_tree.predict(x_test)
