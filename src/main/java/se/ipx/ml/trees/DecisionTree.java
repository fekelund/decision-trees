package se.ipx.ml.trees;

import java.util.List;

import se.ipx.ml.data.Vector;

/**
 * 
 * @author Fredrik Ekelund
 * 
 * @param <C>
 */
public interface DecisionTree<C> {

	@SuppressWarnings("unchecked")
	C predict(C... featureVector);

	C predict(Vector<C> featureVector);

	C predict(List<C> featureVector);

}
