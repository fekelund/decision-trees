package se.ipx.ml.trees.regression;

import static se.ipx.ml.util.Util.asMatrix;
import static se.ipx.ml.util.Util.asScalar;
import static se.ipx.ml.util.Util.multiply;
import static se.ipx.ml.util.Util.transpose;

import org.ojalgo.matrix.BasicMatrix;
import org.ojalgo.matrix.PrimitiveMatrix;

import se.ipx.ml.Instances;

public class ModelTree extends AbstractRegressionTree {

	private ModelTree(Node root, int numFeatures, String targetLabel, String[] featureLabels) {
		super(root, numFeatures, targetLabel, featureLabels);
	}
	
	public static Trainer newTrainer() {
		return new Trainer();
	}
	
	static class ModelLeafNode extends Node {

		final double[][] ws;
		
		ModelLeafNode(double[] ws) {
			this.ws = asMatrix(ws);
		}
		
		@Override
		double getValue(double[] features) {
			return asScalar(multiply(ws, transpose(features)));
		}
		
		@Override
		void write(StringBuilder sb, int depth, String prefix, String[] labels) {
			for (int i = 0; i < depth; i++) {
				sb.append("|   ");
			}

			if (prefix != null) {
				sb.append(prefix);
			}

			sb.append(" : ").append(ws[0]).append('\n');
		}
	}
	
	public static class Trainer extends AbstractTrainer {

		private Instances trainingSet;
		private double minError;
		private int minRowsInSplit;
		
		public Trainer() {
			minError = 0.001D;
			minRowsInSplit = 3;
		}
		
		public Trainer setMinError(double minSquaredError) {
			this.minError = minSquaredError;
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

		public void validate() {
			if (trainingSet == null) {
				throw new IllegalStateException("Missing training set");
			}
		}
		
		@Override
		protected Node createLeafNode(final Instances set) {
			BasicMatrix X = PrimitiveMatrix.FACTORY.rows(set.getFeatureVectors());
			BasicMatrix y = PrimitiveMatrix.FACTORY.columns(set.getTargetValues());
			BasicMatrix ws = X.solve(y);
			double[] w = new double[ws.getRowDim()];
			for (int i = 0; i < w.length; i++) {
				w[i] = ws.doubleValue(i, 0);
			}

			return new ModelLeafNode(w);
		}
		
		@Override
		protected double getError(final Instances set) {
			BasicMatrix A = PrimitiveMatrix.FACTORY.rows(set.getFeatureVectors());
			BasicMatrix b = PrimitiveMatrix.FACTORY.columns(set.getTargetValues());
			BasicMatrix ws = A.solve(b);
			BasicMatrix yHat = A.multiplyRight(ws);
			BasicMatrix s = b.subtract(yHat);
			s = s.multiplyElements(s);
			double sum = 0D;
			final int m = s.getRowDim();
			final int n = s.getColDim();
			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {
					sum += s.doubleValue(i, j);
				}
			}
			
			return sum;
		}
		
		public ModelTree train() {
			validate();
			return new ModelTree(
					buildTree(trainingSet, minError, minRowsInSplit),
					trainingSet.getNumFeatures(), 
					trainingSet.getTargetLabel(),
					trainingSet.getFeatureLabels());
		}
	}	
}
