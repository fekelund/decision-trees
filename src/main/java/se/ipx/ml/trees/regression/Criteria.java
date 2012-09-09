package se.ipx.ml.trees.regression;

import se.ipx.ml.SplitCriteria;

class Criteria implements SplitCriteria {

	private final int featureIndex;
	private final double featureValue;

	private Criteria(int featureIndex, double featureValue) {
		this.featureIndex = featureIndex;
		this.featureValue = featureValue;
	}
	
	static Criteria basedOn(int featureIndex, double featureValue) {
		return new Criteria(featureIndex, featureValue);
	}

	@Override
	public boolean isLeft(final double[] featureVector) {
		return featureVector[featureIndex] >= featureValue;
	}

	@Override
	public boolean isRight(final double[] featureVector) {
		return featureVector[featureIndex] < featureValue;
	}
}