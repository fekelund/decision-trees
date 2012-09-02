package se.ipx.ml.trees.regression;

import java.util.List;

import se.ipx.ml.Instances;
import se.ipx.ml.SplitCriteria;
import se.ipx.ml.util.Pair;
import se.ipx.ml.util.Util;
import it.unimi.dsi.fastutil.doubles.DoubleOpenHashSet;
import it.unimi.dsi.fastutil.doubles.DoubleSet;

public class RegressionTreeImpl implements RegressionTree {

	private final Node root;
	private final String targetLabel;
	private final String[] featureLabels;
	private final int numFeatures;
	
	private RegressionTreeImpl(Node root, int numFeatures, String targetLabel, String[] featureLabels) {
		this.root = root;
		this.numFeatures = numFeatures;
		this.featureLabels = featureLabels;
		this.targetLabel = targetLabel;
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
	
	public double predict(double[] featureVector) {
		preCheck(featureVector);
		return root.getValue(featureVector);
	}


	public Double predict(Number... featureVector) {
		preCheck(featureVector);
		return root.getValue(Util.convert(featureVector));
	}

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
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		root.write(sb, -1, null, featureLabels);
		return sb.toString();
	}
	
	/**
	 * 
	 * @return
	 */
	public static Builder newBuilder() {
		return new Builder();
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
	
	static class LeafNode extends Node {
		
		final double value;
		
		LeafNode(double value) {
			this.value = value;
		}
		
		@Override
		double getValue(final double[] features) {
			return value;
		}

		@Override
		void write(StringBuilder sb, int depth, String prefix, String[] labels) {
			for (int i = 0; i < depth; i++) {
				sb.append("|   ");				
			}
			
			if (prefix != null) {				
				sb.append(prefix);
			}
			
			sb.append(" : ").append(String.format("%.2f", value)).append(" (X/Y) [A/B]").append('\n');
		}
	}
	
	static class Criteria implements SplitCriteria {

		private int featureIndex;
		private double featureValue;
		
		Criteria use(int featureIndex, double featureValue) {
			this.featureIndex = featureIndex;
			this.featureValue = featureValue;
			return this;
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
	
	public static class Builder {
		
		private static final Criteria criteria = new Criteria();

		private Instances trainingSet;
		private double minSquaredError = 0.001D;
		private int minRowsInSplit = 3;
		
		public Builder setMinSquaredError(double minSquaredError) {
			this.minSquaredError = minSquaredError;
			return this;
		}
		
		public Builder setMinRowsInSplit(int minRowsInSplit) {
			this.minRowsInSplit = minRowsInSplit;
			return this;
		}
		
		public Builder setTrainingSet(Instances instances) {
			this.trainingSet = instances;
			return this;
		}
		
		public void validate() {
			if (trainingSet == null) {
				throw new IllegalStateException("Missing data set");
			}
			
			if (minRowsInSplit < 1) {
				throw new IllegalStateException("The minimum number of rows must be at least 1");				
			}
		}
		
		public RegressionTreeImpl build() {
			Node root = buildTree(trainingSet, minSquaredError, minRowsInSplit);
			return new RegressionTreeImpl(root, trainingSet.getNumFeatures(), trainingSet.getTargetLabel(), trainingSet.getFeatureLabels());
		}

		private static final Node buildTree(final Instances instances, final double minSquaredError, final int minRowsInSplit) {
		    Pair<Integer, ?> split = chooseBestSplit(instances, minSquaredError, minRowsInSplit);
		    Integer feature = split.getLeft();
		    if (feature == null) {
		    	return (Node) split.getRight();
		    }
		    
		    double value = ((Double) split.getRight()).doubleValue();
		    Pair<Instances, Instances> pair = instances.binarySplitOn(criteria.use(feature, value));
		    Node lChild = buildTree(pair.getLeft(), minSquaredError, minRowsInSplit);
		    Node rChild = buildTree(pair.getRight(), minSquaredError, minRowsInSplit);
		    return new InternalNode(lChild, rChild, feature, value);
		}

		private static final Pair<Integer, ?> chooseBestSplit(final Instances instances, final double minSquaredError, final int minRowsInSplit) {
			DoubleSet uniqueTargetValues = new DoubleOpenHashSet(instances.getTargets());
			if (uniqueTargetValues.size() == 1) {			
				return Pair.with(null, createLeafNode(instances));                    			
			}
			
			final double squaredError = getSquaredError(instances);
			double bestSquaredError = Double.POSITIVE_INFINITY; 
			double bestValue = 0D;
			int bestFeature = 0; 
			for (int feature = 0; feature < instances.getNumFeatures(); feature++) {
				double[] featureValues = instances.getFeatureValues(feature);
				for (double value : new DoubleOpenHashSet(featureValues)) {
					Pair<Instances, Instances> pair = instances.binarySplitOn(criteria.use(feature, value));
					Instances l = pair.getLeft();
					Instances r = pair.getRight();
					if (l.getNumInstances() < minRowsInSplit || 
						r.getNumInstances() < minRowsInSplit) {
						continue;
					}
					
					double newSquaredError = getSquaredError(l) + getSquaredError(r);
					if (newSquaredError < bestSquaredError) {
						bestSquaredError = newSquaredError;
						bestFeature = feature;
						bestValue = value;
					}
				}
			}		
			
			if ((squaredError - bestSquaredError) < minSquaredError) {
				return Pair.with(null, createLeafNode(instances));                    
			}

			Pair<Instances, Instances> pair = instances.binarySplitOn(criteria.use(bestFeature, bestValue));
			Instances l = pair.getLeft();
			Instances r = pair.getRight();
			if (l.getNumInstances() < minRowsInSplit || 
				r.getNumInstances() < minRowsInSplit) {				
				return Pair.with(null, createLeafNode(instances));
			}
			
			return Pair.with(bestFeature, bestValue);
		}
		
		private static final LeafNode createLeafNode(final Instances instances) {
			return new LeafNode(Util.mean(instances.getTargets()));
		}
		
		private static final double getSquaredError(final Instances instances) {
			return Util.variance(instances.getTargets()) * instances.getNumInstances();
		}
	}
}
