package se.ipx.ml.trees.regression;

import se.ipx.ml.trees.DecisionTree;

public interface RegressionTree extends DecisionTree<Double> {

	double predict(double[] featureVector);
	
}
