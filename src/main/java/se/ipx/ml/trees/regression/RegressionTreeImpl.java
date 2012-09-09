package se.ipx.ml.trees.regression;

import static se.ipx.ml.util.Util.mean;
import static se.ipx.ml.util.Util.variance;

import java.util.concurrent.ForkJoinPool;

import se.ipx.ml.Instances;

public class RegressionTreeImpl extends AbstractRegressionTree {

	private RegressionTreeImpl(Node root, int numFeatures, String targetLabel, String[] featureLabels) {
		super(root, numFeatures, targetLabel, featureLabels);
	}

	public static Trainer newTrainer() {
		return new Trainer();
	}
	
	static class RegressionLeafNode extends Node {

		final double value;

		RegressionLeafNode(double value) {
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

			sb.append(" : ").append(String.format("%.2f", value)).append('\n');
		}
	}

	public static class Trainer extends AbstractTrainer {
		
		private ForkJoinPool pool;
		private Instances trainingSet;
		private double minError;
		private int minRowsInSplit;
		private int numThreads;
		
		public Trainer() {
			minError = 0.001D;
			minRowsInSplit = 3;
		}
		
		public Trainer setMinError(double minError) {
			this.minError = minError;
			return this;
		}

		public Trainer setMinRowsInSplit(int minRowsInSplit) {
			if (minRowsInSplit < 1) {
				throw new IllegalStateException();
			}
			
			this.minRowsInSplit = minRowsInSplit;
			return this;
		}

		public Trainer setTrainingSet(Instances trainingSet) {
			this.trainingSet = trainingSet;
			return this;
		}

		public Trainer setForkJoinPool(ForkJoinPool pool) {
			this.pool = pool;
			return this;
		}
		
		public Trainer setNumThreads(int numThreads) {
			this.numThreads = numThreads;
			return this;
		}
		
		public void validate() {
			if (trainingSet == null) {
				throw new IllegalStateException("Missing training set");
			}
		}
		
		@Override
		protected Node createLeafNode(final Instances set) {
			return new RegressionLeafNode(mean(set.getTargetValues()));
		}
		
		@Override
		protected double getError(final Instances set) {
			return variance(set.getTargetValues()) * set.getNumInstances();
		}
		
		public RegressionTreeImpl train() {
			validate();
			if (pool == null) {
				pool = new ForkJoinPool(numThreads);
			}
			
			Node root = pool.invoke(new TreeBuildingTask(trainingSet, minError, minRowsInSplit));
			return new RegressionTreeImpl(root,
					trainingSet.getNumFeatures(), 
					trainingSet.getTargetLabel(),
					trainingSet.getFeatureLabels());
		}
	}
}
