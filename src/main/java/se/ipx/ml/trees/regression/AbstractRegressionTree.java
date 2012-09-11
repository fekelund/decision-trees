package se.ipx.ml.trees.regression;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.concurrent.RecursiveTask;

import se.ipx.ml.data.Instances;
import se.ipx.ml.data.Vector;
import se.ipx.ml.trees.DecisionTree;
import se.ipx.ml.util.Pair;
import se.ipx.ml.util.Util;

/**
 * 
 * @author Fredrik Ekelund
 * 
 */
public abstract class AbstractRegressionTree implements DecisionTree<Double> {

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

	public double predict(double[] featureVector) {
		preCheck(featureVector);
		return root.getValue(featureVector);
	}

	@Override
	public Double predict(Double... featureVector) {
		preCheck(featureVector);
		return root.getValue(Util.convert(featureVector));
	}

	@Override
	public Double predict(List<Double> featureVector) {
		preCheck(featureVector);
		return root.getValue(Util.convert(featureVector));
	}

	@Override
	public Double predict(Vector<Double> featureVector) {
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

	private void preCheck(final Vector<?> vector) {
		if (vector == null) {
			throw new NullPointerException();
		}

		final int n = vector.getLength();
		if (n != numFeatures) {
			throw new IllegalArgumentException();
		}

		for (int i = 0; i < n; i++) {
			if (vector.getValue(i) == null) {
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
		final Double value;
		final int feature;

		InternalNode(Node left, Node right, int feature, Double value) {
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

	static abstract class AbstractTrainer implements Serializable {

		private static final long serialVersionUID = 1L;

		protected abstract Node createLeafNode(Instances<Double> instances);

		protected abstract double getError(Instances<Double> instances);

		class TreeBuildingTask extends RecursiveTask<Node> {

			private static final long serialVersionUID = 1L;

			final Instances<Double> set;
			final double minError;
			final int minRows;

			TreeBuildingTask(Instances<Double> set, double minError, int minRowsInSplit) {
				this.set = set;
				this.minError = minError;
				this.minRows = minRowsInSplit;
			}

			@Override
			protected Node compute() {
				Pair<Integer, ?> split = chooseBestSplit(set);
				Integer feature = split.getLeft();
				if (feature == null) {
					return (Node) split.getRight();
				}

				Double value = (Double) split.getRight();
				Pair<Instances<Double>, Instances<Double>> sets = set.splitUsing(Criteria.basedOn(feature, value));
				TreeBuildingTask leftBranch = new TreeBuildingTask(sets.getLeft(), minError, minRows);
				leftBranch.fork();
				TreeBuildingTask rightBranch = new TreeBuildingTask(sets.getRight(), minError, minRows);
				Node rightChild = rightBranch.compute();
				Node leftChild = leftBranch.join();
				return new InternalNode(leftChild, rightChild, feature, value);
			}

			protected Pair<Integer, ?> chooseBestSplit(final Instances<Double> set) {
				if (set.getTargets().getUniqueValues().size() == 1) {
					return Pair.with(null, createLeafNode(set));
				}

				final double error = getError(set);
				final SortedSet<ErrorCalculationResult> sortedResults = new TreeSet<ErrorCalculationResult>();
				for (int feature = 0; feature < set.getNumFeatures(); feature++) {
					Set<Double> values = set.getFeatures(feature).getUniqueValues();
					List<ErrorCalculationTask> tasks = new ArrayList<ErrorCalculationTask>(values.size());
					for (Double value : values) {
						tasks.add(new ErrorCalculationTask(set, minRows, feature, value));
					}

					invokeAll(tasks);
					for (ErrorCalculationTask task : tasks) {
						try {
							ErrorCalculationResult result = task.get();
							if (result != null) {
								sortedResults.add(result);
							}
						} catch (Exception e) {
							throw new RuntimeException(e);
						}
					}
				}

				if (sortedResults.isEmpty()) {
					return Pair.with(null, createLeafNode(set));
				}

				ErrorCalculationResult best = sortedResults.first();
				if ((error - best.error) < minError) {
					return Pair.with(null, createLeafNode(set));
				}

				Pair<Instances<Double>, Instances<Double>> sets = set.splitUsing(Criteria.basedOn(best.feature,
						best.value));
				if (sets.getLeft().getNumInstances() < minRows || sets.getRight().getNumInstances() < minRows) {
					return Pair.with(null, createLeafNode(set));
				}

				return Pair.with(best.feature, best.value);
			}
		
		}

		class ErrorCalculationTask extends RecursiveTask<ErrorCalculationResult> {

			private static final long serialVersionUID = 1L;

			final Instances<Double> set;
			final int minRows;
			final int feature;
			final Double value;

			ErrorCalculationTask(Instances<Double> set, int minRowsInSplit, int feature, Double value) {
				this.set = set;
				this.minRows = minRowsInSplit;
				this.feature = feature;
				this.value = value;
			}

			@Override
			protected ErrorCalculationResult compute() {
				Pair<Instances<Double>, Instances<Double>> sets = set.splitUsing(Criteria.basedOn(feature, value));
				Instances<Double> lSet = sets.getLeft();
				Instances<Double> rSet = sets.getRight();
				if (lSet.getNumInstances() < minRows || rSet.getNumInstances() < minRows) {
					return null;
				}

				return ErrorCalculationResult.from(getError(lSet) + getError(rSet), feature, value);
			}
		
		}
	
	}

	static class ErrorCalculationResult implements Comparable<ErrorCalculationResult> {

		final double error;
		final Double value;
		final int feature;

		ErrorCalculationResult(double error, int feature, Double value) {
			this.error = error;
			this.value = value;
			this.feature = feature;
		}

		static ErrorCalculationResult from(Double error, int feature, Double value) {
			return new ErrorCalculationResult(error, feature, value);
		}

		@Override
		public int compareTo(ErrorCalculationResult o) {
			double res = error - o.error;
			if (res < 0D) {
				return -1;
			} else if (res > 0D) {
				return 1;
			} else {
				return 0;
			}
		}

		@Override
		public int hashCode() {
			int hash = 7;
			final long bits = Double.doubleToLongBits(error);
			hash = hash * 31 + (int) (bits ^ (bits >>> 32));
			hash = hash * 31 + value.hashCode();
			hash = hash * 31 + feature;
			return hash;
		}

		@Override
		public boolean equals(Object obj) {
			if (obj == null) {
				return false;
			}

			if (!(obj instanceof ErrorCalculationResult)) {
				return false;
			}

			ErrorCalculationResult that = (ErrorCalculationResult) obj;
			return this.feature == that.feature && this.error == that.error && this.value.equals(that.value);
		}
	
	}
	
}
