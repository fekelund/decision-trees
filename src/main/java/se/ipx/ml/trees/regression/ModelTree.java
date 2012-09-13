/**
 * Copyright (C) 2012 Fredrik Ekelund <fredrik@ipx.se>
 *
 * This file is part of Decision Trees.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package se.ipx.ml.trees.regression;

import static se.ipx.ml.util.Util.asMatrix;
import static se.ipx.ml.util.Util.asScalar;
import static se.ipx.ml.util.Util.multiply;
import static se.ipx.ml.util.Util.transpose;

import java.util.concurrent.ForkJoinPool;

import org.ojalgo.matrix.BasicMatrix;
import org.ojalgo.matrix.PrimitiveMatrix;

import se.ipx.ml.data.Instances;

/**
 * 
 * @author Fredrik Ekelund
 * 
 */
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

			sb.append(" : ");
			boolean first = true;
			for (double d : ws[0]) {
				if (first) {
					sb.append("[");
					first = false;
				} else {
					sb.append(", ");
				}

				sb.append(d);
			}

			sb.append("]\n");
		}
		
	}

	public static class Trainer extends AbstractTrainer {

		private static final long serialVersionUID = 1L;

		private transient ForkJoinPool pool;
		private Instances<Double> set;
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

		public Trainer setTrainingSet(Instances<Double> set) {
			this.set = set;
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
			if (set == null) {
				throw new IllegalStateException("Missing training set");
			}
		}

		@Override
		protected Node createLeafNode(final Instances<Double> set) {
			BasicMatrix X = PrimitiveMatrix.FACTORY.copy(set.getFeatureMatrix());
			BasicMatrix y = PrimitiveMatrix.FACTORY.columns(set.getTargets());
			BasicMatrix ws = X.solve(y);
			double[] w = new double[ws.getRowDim()];
			for (int i = 0; i < w.length; i++) {
				w[i] = ws.doubleValue(i, 0);
			}

			return new ModelLeafNode(w);
		}

		@Override
		protected double getError(final Instances<Double> set) {
			BasicMatrix A = PrimitiveMatrix.FACTORY.copy(set.getFeatureMatrix());
			BasicMatrix b = PrimitiveMatrix.FACTORY.columns(set.getTargets());
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
			if (pool == null) {
				pool = new ForkJoinPool(numThreads);
			}

			Node root = pool.invoke(new TreeBuildingTask(set, minError, minRowsInSplit));
			return new ModelTree(root, set.getNumFeatures(), set.getTargetLabel(), set.getFeatureLabels());
		}

	}

}
