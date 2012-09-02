package se.ipx.ml;

import se.ipx.ml.util.Pair;

public interface Instances {

	double[][] getFeatureVectors();
	
	double[] getFeatureVector(int instanceIndex);
	
	double[] getFeatureValues(int featureIndex);
	
	double[] getTargets();
	
	int getNumInstances();
	
	int getNumFeatures();
	
	String[] getFeatureLabels();
	
	String getFeatureLabel(int featureIndex);
	
	String getTargetLabel();
	
	Pair<Instances, Instances> binarySplitOn(SplitCriteria criteria);
	
}
