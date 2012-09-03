package se.ipx.ml.trees;

import java.util.List;

/**
 * 
 * @author Fredrik Ekelund
 * 
 * @param <C>
 */
public interface DecisionTree<C> {

	/**
	 * 
	 * @param featureVector
	 * @return
	 */
	C predict(Number... featureVector);

	/**
	 * 
	 * @param featureVector
	 * @return
	 */
	C predict(List<? extends Number> featureVector);

}
