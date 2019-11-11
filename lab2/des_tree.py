from sklearn import tree, metrics, datasets
import graphviz

def main():
    digits = datasets.load_digits()

    split = int(0.7 * digits.data.shape[0])
    train_feature = digits.data[:split]
    train_label = digits.target[:split]
    test_feature = digits.data[split:]
    test_label = digits.target[split:]

    classifier = tree.DecisionTreeClassifier()
    classifier.fit(train_feature, train_label)

    pred = classifier.predict(test_feature)
    print(pred)

    dot_data = tree.export_graphviz(classifier, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("tree")

if __name__ == "__main__":
    main()