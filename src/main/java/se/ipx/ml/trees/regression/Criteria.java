package se.ipx.ml.trees.regression;

import se.ipx.ml.data.SplitCriteria;
import se.ipx.ml.data.Vector;

class Criteria implements SplitCriteria<Double> {

	private final int feature;
	private final double value;

	private Criteria(final int feature, final double value) {
		this.feature = feature;
		this.value = value;
	}

	static Criteria basedOn(final int feature, final double value) {
		return new Criteria(feature, value);
	}

	@Override
	public boolean isLeft(final Vector<Double> featureVector) {
		return featureVector.getValue(feature).doubleValue() >= value;
	}

	@Override
	public boolean isRight(final Vector<Double> featureVector) {
		return featureVector.getValue(feature).doubleValue() < value;
	}

}