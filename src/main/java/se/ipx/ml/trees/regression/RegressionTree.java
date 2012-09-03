package se.ipx.ml.trees.regression;

import se.ipx.ml.trees.DecisionTree;

/**
 * 
 * @author Fredrik Ekelund
 * 
 */
public interface RegressionTree extends DecisionTree<Double> {

	/**
	 * 
	 * @param featureVector
	 * @return
	 */
	double predict(double[] featureVector);

}
