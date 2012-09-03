package se.ipx.ml;

import it.unimi.dsi.fastutil.doubles.DoubleArrayList;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map.Entry;
import java.util.TreeMap;

import se.ipx.ml.util.Pair;
import se.ipx.ml.util.Util;

public class InstancesImpl implements Instances {

	private final String targetLabel;
	private final String[] fetaureLabels;
	private final double[] targetValues;
	private final double[][] valuesByRow;
	private final double[][] valuesByCol;

	private InstancesImpl(String targetLabel, double[] targetValues, String[] fetaureLabels, double[][] featureValues) {
		this.targetLabel = targetLabel;
		this.targetValues = targetValues;
		this.fetaureLabels = fetaureLabels;
		this.valuesByRow = featureValues;
		this.valuesByCol = Util.transpose(featureValues);
	}

	@Override
	public String[] getFeatureLabels() {
		return fetaureLabels;
	}

	@Override
	public String getTargetLabel() {
		return targetLabel;
	}

	@Override
	public String getFeatureLabel(final int featureIndex) {
		return fetaureLabels[featureIndex];
	}

	@Override
	public double[][] getFeatureVectors() {
		return valuesByRow;
	}

	@Override
	public double[] getFeatureVector(final int instanceIndex) {
		return valuesByRow[instanceIndex];
	}

	@Override
	public double[] getFeatureValues(final int featureIndex) {
		return valuesByCol[featureIndex];
	}

	@Override
	public double[] getTargetValues() {
		return targetValues;
	}

	@Override
	public int getNumInstances() {
		return valuesByRow.length;
	}

	@Override
	public int getNumFeatures() {
		return valuesByCol.length;
	}

	@Override
	public Pair<Instances, Instances> binarySplitOn(final SplitCriteria criteria) {
		Builder l = InstancesImpl.newBuilder().setFeatureLabels(fetaureLabels).setTargetLabel(targetLabel);
		Builder r = InstancesImpl.newBuilder().setFeatureLabels(fetaureLabels).setTargetLabel(targetLabel);
		for (int i = 0; i < valuesByRow.length; i++) {
			double[] featureVector = valuesByRow[i];
			if (criteria.isLeft(featureVector)) {
				l.addInstance(targetValues[i], featureVector);
			} else {
				r.addInstance(targetValues[i], featureVector);
			}
		}

		return Pair.with(l.build(), r.build());
	}

	public static Builder newBuilder() {
		return new Builder();
	}

	public static class Builder {

		private final List<double[]> featureVectors;
		private final DoubleArrayList targetValues;
		private final TreeMap<Integer, String> featureLabels;
		private String taretLabel;
		private int numFeatures;

		public Builder() {
			featureVectors = new ArrayList<double[]>(1024);
			featureLabels = new TreeMap<Integer, String>();
			targetValues = new DoubleArrayList(1024);
			taretLabel = "";
			numFeatures = -1;
		}

		public Builder addInstance(final double targetValue, final double[] featureVector) {
			if (featureVector == null) {
				throw new NullPointerException();
			}

			if (numFeatures == -1) {
				numFeatures = featureVector.length;
			} else if (numFeatures != featureVector.length) {
				throw new IllegalArgumentException();
			}

			featureVectors.add(featureVector);
			targetValues.add(targetValue);
			return this;
		}

		public Builder addInstance(final Number targetValue, final Number... featureVector) {
			if (featureVector == null || targetValue == null) {
				throw new NullPointerException();
			}

			double[] primitives = new double[featureVector.length];
			for (int i = 0; i < featureVector.length; i++) {
				if (featureVector[i] == null) {
					throw new NullPointerException();
				}

				primitives[i] = featureVector[i].doubleValue();
			}

			return addInstance(targetValue.doubleValue(), primitives);
		}

		public Builder addInstance(final Number targetValue, final List<? extends Number> featureVector) {
			if (featureVector == null || targetValue == null) {
				throw new NullPointerException();
			}

			double[] primitives = new double[featureVector.size()];
			for (int i = 0; i < featureVector.size(); i++) {
				Number value = featureVector.get(i);
				if (value == null) {
					throw new NullPointerException();
				}

				primitives[i] = value.doubleValue();
			}

			return addInstance(targetValue.doubleValue(), primitives);
		}

		public Builder setFeatureLabels(List<? extends CharSequence> featureLabels) {
			if (featureLabels != null) {
				for (int i = 0; i < featureLabels.size(); i++) {
					setFeatureLabel(featureLabels.get(i), i);
				}
			}

			return this;
		}

		public Builder setFeatureLabels(final CharSequence... featureLabels) {
			if (featureLabels != null) {
				for (int i = 0; i < featureLabels.length; i++) {
					setFeatureLabel(featureLabels[i], i);
				}
			}

			return this;
		}

		public Builder setFeatureLabel(final CharSequence featureLabel, int featureIndex) {
			if (featureIndex < 0) {
				throw new IllegalArgumentException();
			}

			if (featureLabel != null) {
				featureLabels.put(Integer.valueOf(featureIndex), featureLabel.toString());
			}

			return this;
		}

		public Builder setTargetLabel(final CharSequence targetLabel) {
			if (targetLabel != null) {
				this.taretLabel = targetLabel.toString();
			}

			return this;
		}

		public void validate() {
			// NO-OP
		}

		String[] getFeatureLabels() {
			String[] labels;
			if (numFeatures > 0) {
				labels = new String[numFeatures];
				for (int i = 0; i < numFeatures; i++) {
					String label = featureLabels.get(i);
					labels[i] = label != null ? label : "";
				}
			} else if (!featureLabels.isEmpty()) {
				Integer len = featureLabels.lastKey();
				labels = new String[len + 1];
				Arrays.fill(labels, "");
				for (Entry<Integer, String> entry : featureLabels.entrySet()) {
					labels[entry.getKey()] = entry.getValue();
				}
			} else {
				labels = new String[0];
			}

			return labels;
		}

		double[][] getFeatureVectors() {
			return featureVectors.toArray(new double[featureVectors.size()][]);
		}

		public Instances build() {
			validate();
			return new InstancesImpl(taretLabel, targetValues.toDoubleArray(), getFeatureLabels(), getFeatureVectors());
		}
	}
}