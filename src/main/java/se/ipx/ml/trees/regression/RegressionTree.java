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

import static se.ipx.ml.util.Util.mean;
import static se.ipx.ml.util.Util.variance;

import java.util.concurrent.ForkJoinPool;

import se.ipx.ml.data.Instances;

/**
 * 
 * @author Fredrik Ekelund
 * 
 */
public class RegressionTree extends AbstractRegressionTree {

	private RegressionTree(Node root, int numFeatures, String targetLabel, String[] featureLabels) {
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

		public Trainer setTrainingSet(Instances<Double> trainingSet) {
			this.set = trainingSet;
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
			return new RegressionLeafNode(mean(set.getTargets()));
		}

		@Override
		protected double getError(final Instances<Double> set) {
			return variance(set.getTargets()) * set.getNumInstances();
		}

		public RegressionTree train() {
			validate();
			if (pool == null) {
				pool = new ForkJoinPool(numThreads);
			}

			Node root = pool.invoke(new TreeBuildingTask(set, minError, minRowsInSplit));
			return new RegressionTree(root, set.getNumFeatures(), set.getTargetLabel(), set.getFeatureLabels());
		}

	}

}
