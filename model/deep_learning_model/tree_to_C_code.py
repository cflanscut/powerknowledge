import numpy as np
from sklearn.tree import DecisionTreeRegressor, _tree
from sklearn.ensemble import RandomForestClassifier
import os

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([2, 0, 2, 1, 0, 3, 1, 0, 0, 1]).reshape(-1, 1)
class_num = 4  # y有四类
# 训练单棵树
dtr = DecisionTreeRegressor()
dtr.fit(x, y)

# 训练森林
rf = RandomForestClassifier(
    n_estimators=2,
    max_depth=5,
)
rf.fit(x, y)


def tree_to_code(single_tree, feature_names, output_file):
    with open(output_file, 'a+') as f:
        parameters = "("
        for idx, value in enumerate(feature_names):
            if idx != 0:
                parameters += ','
            parameters += "float" + " " + str(value)
        parameters += ")"
        f.write("float *tree" + parameters + "\n")
        f.write("{\n")
        f.write("    static float r[{}] = {{0}};\n".format(class_num))
        f.close()

    tree_ = single_tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    # tree内部的节点编号规则为，先左后右，先纵后横，depth只是用来计算缩进值
    def recurse(node, depth):
        indent = "    " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            # 树枝节点
            name = feature_name[node]
            threshold = tree_.threshold[node]
            with open(output_file, 'a+') as f:
                f.write("{}if ({} <= {})\n".format(indent, name,
                                                   round(threshold, 3)))
                f.write("{}".format(indent) + "{\n")
                f.close()
            recurse(tree_.children_left[node], depth + 1)
            with open(output_file, 'a+') as f:
                f.write("{}".format(indent) + "}\n")
                f.close()
            # children_left为当前节点的左子节点编号
            with open(output_file, 'a+') as f:
                f.write("{}else\n".format(indent))
                f.write("{}".format(indent) + "{\n")
                f.close()
            recurse(tree_.children_right[node], depth + 1)
            # children_right为当前节点的右子节点编号
            with open(output_file, 'a+') as f:
                f.write("{}".format(indent) + "}\n")
                f.close()
        else:
            # 叶子节点
            with open(output_file, 'a+') as f:
                s = '{'
                for idx, value in enumerate(tree_.value[node][0]):
                    if idx != 0:
                        s += ','
                    s += str(value)
                s += '}'
                f.write("{}float o[] = ".format(indent) + s + ';\n')
                f.write('{}arrayplus(r,o);\n'.format(indent))
                f.close()

    recurse(0, 1)
    with open(output_file, 'a+') as f:
        f.write("return r;\n")
        f.write("}\n")
        f.close()


def forest_to_code(forest, feature_names, output_file):
    with open(output_file, 'a+') as f:
        parameters = "("
        for idx, value in enumerate(feature_names):
            if idx != 0:
                parameters += ','
            parameters += "float" + " " + str(value)
        parameters += ")"
        f.write("float *tree" + parameters + "\n")
        f.write("{\n")
        f.write("    static float r[{}] = {{0}};\n".format(class_num))
        f.close()

    for idx, estimator in enumerate(forest.estimators_):
        with open(output_file, 'a+') as f:
            f.write("    // No.{} tree\n".format(idx))
            f.close()

        tree_ = estimator.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        # tree内部的节点编号规则为，先左后右，先纵后横，depth只是用来计算缩进值
        def recurse(node, depth):
            indent = "    " * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                # 树枝节点
                name = feature_name[node]
                threshold = tree_.threshold[node]
                with open(output_file, 'a+') as f:
                    f.write("{}if ({} <= {})\n".format(indent, name,
                                                       round(threshold, 3)))
                    f.write("{}".format(indent) + "{\n")
                    f.close()
                recurse(tree_.children_left[node], depth + 1)
                with open(output_file, 'a+') as f:
                    f.write("{}".format(indent) + "}\n")
                    f.close()
                # children_left为当前节点的左子节点编号
                with open(output_file, 'a+') as f:
                    f.write("{}else\n".format(indent))
                    f.write("{}".format(indent) + "{\n")
                    f.close()
                recurse(tree_.children_right[node], depth + 1)
                # children_right为当前节点的右子节点编号
                with open(output_file, 'a+') as f:
                    f.write("{}".format(indent) + "}\n")
                    f.close()
            else:
                # 叶子节点
                with open(output_file, 'a+') as f:
                    s = '{'
                    for idx, value in enumerate(tree_.value[node][0]):
                        if idx != 0:
                            s += ','
                        s += str(value)
                    s += '}'
                    f.write("{}float o[] = ".format(indent) + s + ';\n')
                    f.write('{}arrayplus(r,o);\n'.format(indent))
                    f.close()

        recurse(0, 1)

    with open(output_file, 'a+') as f:
        f.write("return r;\n")
        f.write("}\n")
        f.close()


pwd = os.getcwd()
# tree_to_code(dtr, ['x'], pwd + '/model/deep_learning_model/test_tree.txt')
forest_to_code(rf, ['x'], pwd + '/model/deep_learning_model/test_tree.txt')
