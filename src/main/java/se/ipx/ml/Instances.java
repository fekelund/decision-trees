package se.ipx.ml;

import se.ipx.ml.util.Pair;

/**
 * 
 * @author Fredrik Ekelund
 * 
 */
public interface Instances {

	/**
	 * 
	 * @return
	 */
	double[][] getFeatureVectors();

	/**
	 * 
	 * @param instanceIndex
	 * @return
	 */
	double[] getFeatureVector(int instanceIndex);

	/**
	 * 
	 * @param featureIndex
	 * @return
	 */
	double[] getFeatureValues(int featureIndex);

	/**
	 * 
	 * @return
	 */
	double[] getTargetValues();

	/**
	 * 
	 * @return
	 */
	int getNumInstances();

	/**
	 * 
	 * @return
	 */
	int getNumFeatures();

	/**
	 * 
	 * @return
	 */
	String[] getFeatureLabels();

	/**
	 * 
	 * @param featureIndex
	 * @return
	 */
	String getFeatureLabel(int featureIndex);

	/**
	 * 
	 * @return
	 */
	String getTargetLabel();

	/**
	 * 
	 * @param criteria
	 * @return
	 */
	Pair<Instances, Instances> binarySplitOn(SplitCriteria criteria);

}
