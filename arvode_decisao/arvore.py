from sklearn.datasets import load_diabetes
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn import tree
import graphviz
import pydotplus

diabetes = load_diabetes()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(diabetes.data, diabetes.target)


dot_data = tree.export_graphviz(clf, out_file=None)

#========================================================
graph = graphviz.Source(dot_data)
graph.render("diabetes")
#========================================================


dot_data = StringIO()

tree.export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

graph.write_pdf('tree.pdf')


