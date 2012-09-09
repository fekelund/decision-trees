package se.ipx.ml.trees.regression;

import it.unimi.dsi.fastutil.doubles.DoubleOpenHashSet;
import it.unimi.dsi.fastutil.doubles.DoubleSet;

import java.util.ArrayList;
import java.util.List;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.concurrent.RecursiveTask;

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
				
		class TreeBuildingTask extends RecursiveTask<Node> {

			private static final long serialVersionUID = 1L;
			
			final Instances set;
			final double minError;
			final int minRows;
			
			TreeBuildingTask(Instances set, double minSquaredError, int minRowsInSplit) {
				this.set = set;
				this.minError = minSquaredError;
				this.minRows = minRowsInSplit;
			}

			@Override
			protected Node compute() {
				Pair<Integer, ?> split = chooseBestSplit(set);
				Integer feature = split.getLeft();
				if (feature == null) {
					return (Node) split.getRight();
				}
				
				double value = ((Double) split.getRight()).doubleValue();
				Pair<Instances, Instances> pair = set.splitUsing(Criteria.basedOn(feature, value));
				TreeBuildingTask lBranch = new TreeBuildingTask(pair.getLeft(), minError, minRows);
				TreeBuildingTask rBranch = new TreeBuildingTask(pair.getRight(), minError, minRows);
				lBranch.fork();
				Node rChild = rBranch.compute();
				Node lChild = lBranch.join();
				return new InternalNode(lChild, rChild, feature, value);
			}
			
			protected Pair<Integer, ?> chooseBestSplit(final Instances set) {
				DoubleSet uniqueTargetValues = new DoubleOpenHashSet(set.getTargetValues());
				if (uniqueTargetValues.size() == 1) {
					return Pair.with(null, createLeafNode(set));
				}

				final double error = getError(set);
				final SortedSet<ErrorCalculationResult> sortedResults = new TreeSet<ErrorCalculationResult>();
				for (int feature = 0; feature < set.getNumFeatures(); feature++) {
					double[] values = getUniqueValues(set.getFeatureValues(feature));
					
					
					List<ErrorCalculationTask> tasks = new ArrayList<ErrorCalculationTask>(values.length);
					for (double value : values) {
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

				Pair<Instances, Instances> pair = set.splitUsing(Criteria.basedOn(best.feature, best.value));
				Instances l = pair.getLeft();
				Instances r = pair.getRight();
				if (l.getNumInstances() < minRows || r.getNumInstances() < minRows) {
					return Pair.with(null, createLeafNode(set));
				}

				return Pair.with(best.feature, best.value);
			}
		}
		
		class ErrorCalculationTask extends RecursiveTask<ErrorCalculationResult> {

			private static final long serialVersionUID = 1L;
			
			final Instances set;
			final int minRows;
			final int feature;
			final double value;
			
			ErrorCalculationTask(Instances set, int minRowsInSplit, int feature, double value) {
				this.set = set;
				this.minRows = minRowsInSplit;
				this.feature = feature;
				this.value = value;
			}
			
			@Override
			protected ErrorCalculationResult compute() {
				Pair<Instances, Instances> pair = set.splitUsing(Criteria.basedOn(feature, value));
				Instances l = pair.getLeft();
				Instances r = pair.getRight();
				if (l.getNumInstances() < minRows || r.getNumInstances() < minRows) {
					return null;
				}

				return ErrorCalculationResult.from(getError(l) + getError(r), feature, value);
			}
		}
		
		private static final double[] getUniqueValues(final double[] values) {
			return new DoubleOpenHashSet(values).toDoubleArray();
		}
	}
	
	static class ErrorCalculationResult implements Comparable<ErrorCalculationResult> {
		
		final Double error;
		final Double value;
		final int feature;
		
		ErrorCalculationResult(double error, int feature, double value) {
			this.error = Double.valueOf(error);
			this.value = Double.valueOf(value);
			this.feature = feature;
		}
		
		static ErrorCalculationResult from(double error, int feature, double value) {
			return new ErrorCalculationResult(error, feature, value);
		}

		@Override
		public int compareTo(ErrorCalculationResult o) {
			double res = error.doubleValue() - o.error.doubleValue();
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
			hash = hash * 31 + error.hashCode();
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
			return (this.error == that.error) && (this.value == that.value) && (this.feature == that.feature);
		}
	}
}
