package se.ipx.ml.trees.regression;

import it.unimi.dsi.fastutil.doubles.DoubleOpenHashSet;
import it.unimi.dsi.fastutil.doubles.DoubleSet;

import java.util.List;

import se.ipx.ml.Instances;
import se.ipx.ml.util.Pair;
import se.ipx.ml.util.Util;

/**
 * 
 * @author Fredrik Ekelund
 * 
 */
public abstract class AbstractRegressionTree implements RegressionTree {

	private final Node root;
	private final String targetLabel;
	private final String[] featureLabels;
	private final int numFeatures;

	AbstractRegressionTree(Node root, int numFeatures, String targetLabel, String[] featureLabels) {
		this.root = root;
		this.numFeatures = numFeatures;
		this.featureLabels = featureLabels;
		this.targetLabel = targetLabel;
	}

	@Override
	public double predict(double[] featureVector) {
		preCheck(featureVector);
		return root.getValue(featureVector);
	}

	@Override
	public Double predict(Number... featureVector) {
		preCheck(featureVector);
		return root.getValue(Util.convert(featureVector));
	}

	@Override
	public Double predict(List<? extends Number> featureVector) {
		preCheck(featureVector);
		return root.getValue(Util.convert(featureVector));
	}

	public String getTargetLabel() {
		return targetLabel;
	}

	public String[] getFeatureLabels() {
		return featureLabels;
	}

	public int getNumFeatures() {
		return numFeatures;
	}

	private void preCheck(final double[] vector) {
		if (vector == null) {
			throw new NullPointerException();
		}

		if (vector.length != numFeatures) {
			throw new IllegalArgumentException();
		}
	}

	private void preCheck(final Object[] vector) {
		if (vector == null) {
			throw new NullPointerException();
		}

		if (vector.length != numFeatures) {
			throw new IllegalArgumentException();
		}

		for (int i = 0; i < vector.length; i++) {
			if (vector[i] == null) {
				throw new NullPointerException();
			}
		}
	}

	private void preCheck(final List<?> vector) {
		if (vector == null) {
			throw new NullPointerException();
		}

		final int n = vector.size();
		if (n != numFeatures) {
			throw new IllegalArgumentException();
		}

		for (int i = 0; i < n; i++) {
			if (vector.get(i) == null) {
				throw new NullPointerException();
			}
		}
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		root.write(sb, -1, null, featureLabels);
		return sb.toString();
	}
	
	static abstract class Node {

		abstract double getValue(final double[] features);

		abstract void write(final StringBuilder builder, final int depth, final String prefix, final String[] labels);

	}

	static class InternalNode extends Node {

		final Node left;
		final Node right;
		final double value;
		final int feature;

		InternalNode(Node left, Node right, int feature, double value) {
			this.left = left;
			this.right = right;
			this.value = value;
			this.feature = feature;
		}

		@Override
		double getValue(final double[] features) {
			Node child = features[feature] >= value ? left : right;
			return child.getValue(features);
		}

		@Override
		void write(StringBuilder sb, int depth, String prefix, String[] labels) {
			for (int i = 0; i < depth; i++) {
				sb.append("|   ");
			}

			if (prefix != null) {
				sb.append(prefix).append("\n");
			}

			right.write(sb, depth + 1, labels[feature] + " < " + value, labels);
			left.write(sb, depth + 1, labels[feature] + " >= " + value, labels);
		}
	}
		
	static abstract class AbstractTrainer {
		
		protected abstract Node createLeafNode(Instances instances);
		
		protected abstract double getError(Instances instances);
		
		protected final Node buildTree(final Instances set, final double minError, final int minRows) {
			// TODO: ugly, fix!
			Pair<Integer, ?> split = chooseBestSplit(set, minError, minRows);
			Integer feature = split.getLeft();
			if (feature == null) {
				return (Node) split.getRight();
			}
			
			double value = ((Double) split.getRight()).doubleValue();
			Pair<Instances, Instances> pair = set.splitUsing(Criteria.basedOn(feature, value));
			Node lChild = buildTree(pair.getLeft(), minError, minRows);
			Node rChild = buildTree(pair.getRight(), minError, minRows);
			return new InternalNode(lChild, rChild, feature, value);
		}
		
		private final Pair<Integer, ?> chooseBestSplit(final Instances set, final double minError, final int minRows) {
			DoubleSet uniqueTargetValues = new DoubleOpenHashSet(set.getTargetValues());
			if (uniqueTargetValues.size() == 1) {
				return Pair.with(null, createLeafNode(set));
			}

			final double error = getError(set);
			double bestError = Double.POSITIVE_INFINITY;
			double bestValue = 0D;
			int bestFeature = 0;
			for (int feature = 0; feature < set.getNumFeatures(); feature++) {
				double[] values = set.getFeatureValues(feature);
				for (double value : new DoubleOpenHashSet(values)) {
					Pair<Instances, Instances> pair = set.splitUsing(Criteria.basedOn(feature, value));
					Instances l = pair.getLeft();
					Instances r = pair.getRight();
					if (l.getNumInstances() < minRows || r.getNumInstances() < minRows) {
						continue;
					}

					double newError = getError(l) + getError(r);
					if (newError < bestError) {
						bestError = newError;
						bestFeature = feature;
						bestValue = value;
					}
				}
			}

			if ((error - bestError) < minError) {
				return Pair.with(null, createLeafNode(set));
			}

			Pair<Instances, Instances> pair = set.splitUsing(Criteria.basedOn(bestFeature, bestValue));
			Instances l = pair.getLeft();
			Instances r = pair.getRight();
			if (l.getNumInstances() < minRows || r.getNumInstances() < minRows) {
				return Pair.with(null, createLeafNode(set));
			}

			return Pair.with(bestFeature, bestValue);
		}
	}
}
