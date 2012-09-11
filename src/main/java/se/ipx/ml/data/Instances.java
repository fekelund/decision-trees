package se.ipx.ml.data;

import java.io.Serializable;

import se.ipx.ml.util.Pair;

/**
 * 
 * @author Fredrik Ekelund
 * 
 * @param <T>
 */
public interface Instances<T> extends Serializable {

	Pair<Instances<T>, Instances<T>> splitUsing(SplitCriteria<T> criteria);

	Matrix<T> getFeatureMatrix();

	Vector<T> getFeatureVector(int index);

	Vector<T> getFeatures(int index);

	Vector<T> getTargets();

	int getNumInstances();

	int getNumFeatures();

	String getTargetLabel();

	String getFeatureLabel(int index);

	String[] getFeatureLabels();

}
