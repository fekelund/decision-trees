package se.ipx.ml.trees;

import java.util.List;

public interface DecisionTree<C> {
	
	C predict(Number... featureVector);

	C predict(List<? extends Number> featureVector);
	
}
