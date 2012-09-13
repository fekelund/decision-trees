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
package se.ipx.ml.data.impl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.Map.Entry;

import se.ipx.ml.data.Instance;
import se.ipx.ml.data.Instances;
import se.ipx.ml.data.Matrix;
import se.ipx.ml.data.SplitCriteria;
import se.ipx.ml.data.Vector;
import se.ipx.ml.util.Pair;

public class InstancesImpl<T> implements Instances<T> {

	private static final long serialVersionUID = 1L;

	private final T[] features;
	private final Vector<T> targets;

	private final String[] featureLabels;
	private final String targetLabel;

	private final int numRows;
	private final int numCols;

	InstancesImpl(T[] featureValues, List<T> targetValues, String[] featureLabels, String targetLabel, int numRows,
			int numCols) {
		this.features = featureValues;
		this.targets = new TargetColumn<T>(targetValues);

		this.featureLabels = featureLabels;
		this.targetLabel = targetLabel;

		this.numRows = numRows;
		this.numCols = numCols;
	}
	
	@Override
	public Instance<T> getInstance(int index) {
		return new InstanceImpl<T>(this, targets.getValue(index), index);
	}
	
	@Override
	public Matrix<T> getFeatureMatrix() {
		return new FeatureMatrix<T>(features, 0, numRows, numCols);
	}

	@Override
	public Vector<T> getFeatureVector(final int index) {
		return new FeatureRow<T>(features, index * numCols, numCols);
	}

	@Override
	public Vector<T> getFeatures(final int index) {
		return new FeatureColumn<T>(features, index, numCols);
	}

	@Override
	public Vector<T> getTargets() {
		return targets;
	}

	@Override
	public int getNumInstances() {
		return numRows;
	}

	@Override
	public int getNumFeatures() {
		return numCols;
	}

	@Override
	public String[] getFeatureLabels() {
		return featureLabels;
	}

	@Override
	public String getFeatureLabel(final int index) {
		return featureLabels[index];
	}

	@Override
	public String getTargetLabel() {
		return targetLabel;
	}

	@Override
	public Pair<Instances<T>, Instances<T>> splitUsing(final SplitCriteria<T> criteria) {
		final List<Vector<T>> lRows = new ArrayList<Vector<T>>(numRows);
		final List<Vector<T>> rRows = new ArrayList<Vector<T>>(numRows);
		final List<T> lTargets = new ArrayList<T>(numRows);
		final List<T> rTargets = new ArrayList<T>(numRows);
		for (int i = 0, j = 0; i < features.length; i += numCols, j++) {
			final Vector<T> row = new FeatureRow<T>(features, i, numCols);
			if (criteria.isLeft(row)) {
				lRows.add(row);
				lTargets.add(targets.getValue(j));
			} else {
				rRows.add(row);
				rTargets.add(targets.getValue(j));
			}
		}

		final Instances<T> l = new NestedInstances<T>(lRows, lTargets, this);
		final Instances<T> r = new NestedInstances<T>(rRows, rTargets, this);
		return Pair.with(l, r);
	}

	public static <T> Builder<T> newBuilder() {
		return new Builder<T>();
	}

	static class NestedFeatureMatrix<T> implements Matrix<T> {

		private static final long serialVersionUID = 1L;

		private final List<Vector<T>> vectors;
		private final int numCols;

		NestedFeatureMatrix(List<Vector<T>> vectors, final int numCols) {
			this.vectors = vectors;
			this.numCols = numCols;
		}

		public T getValue(final int row, final int col) {
			return vectors.get(row).getValue(col);
		}

		public int getNumRows() {
			return vectors.size();
		}

		public int getNumCols() {
			return numCols;
		}

		@Override
		public double doubleValue(int aRow, int aCol) {
			return get(aRow, aCol).doubleValue();
		}

		@Override
		public int getColDim() {
			return numCols;
		}

		@Override
		public int getRowDim() {
			return vectors.size();
		}

		@Override
		public int size() {
			return vectors.size() * numCols;
		}

		@Override
		public Number get(int aRow, int aCol) {
			T value = getValue(aRow, aCol);
			if (!(value instanceof Number)) {
				throw new UnsupportedOperationException();
			}

			return (Number) value;
		}

	}

	static class NestedInstances<T> implements Instances<T> {

		private static final long serialVersionUID = 1L;

		private final List<Vector<T>> fVectors;
		private final Vector<T> tVector;
		private final String[] fLabels;
		private final String tLabel;
		private final int numCols;

		NestedInstances(List<Vector<T>> fVectors, List<T> tVector, Instances<?> i) {
			this(fVectors, tVector, i.getFeatureLabels(), i.getTargetLabel(), i.getNumFeatures());
		}

		NestedInstances(List<Vector<T>> fVectors, List<T> tVector, String[] fLabels, String tLabel, int numCols) {
			if (fVectors.size() != tVector.size()) {
				throw new IllegalArgumentException();
			}

			this.fVectors = fVectors;
			this.fLabels = fLabels;
			this.tVector = new TargetColumn<T>(tVector);
			this.tLabel = tLabel;
			this.numCols = numCols;
		}

		@Override
		public Instance<T> getInstance(int index) {
			return new InstanceImpl<T>(this, tVector.getValue(index), index);
		}
		
		@Override
		public Matrix<T> getFeatureMatrix() {
			return new NestedFeatureMatrix<T>(fVectors, numCols);
		}

		@Override
		public Vector<T> getFeatureVector(int index) {
			return fVectors.get(index);
		}

		@Override
		public Vector<T> getFeatures(int index) {
			return new NestedFeatureColumn<T>(fVectors, index);
		}

		@Override
		public Vector<T> getTargets() {
			return tVector;
		}

		@Override
		public int getNumInstances() {
			return fVectors.size();
		}

		@Override
		public int getNumFeatures() {
			return numCols;
		}

		@Override
		public String[] getFeatureLabels() {
			return fLabels;
		}

		@Override
		public String getFeatureLabel(int index) {
			return fLabels[index];
		}

		@Override
		public String getTargetLabel() {
			return tLabel;
		}

		@Override
		public Pair<Instances<T>, Instances<T>> splitUsing(SplitCriteria<T> criteria) {
			final int N = fVectors.size();
			final List<Vector<T>> lfv = new ArrayList<Vector<T>>(N);
			final List<Vector<T>> rfv = new ArrayList<Vector<T>>(N);
			final List<T> lt = new ArrayList<T>(N);
			final List<T> rt = new ArrayList<T>(N);
			for (int i = 0; i < N; i++) {
				final Vector<T> fv = fVectors.get(i);
				if (criteria.isLeft(fv)) {
					lfv.add(fv);
					lt.add(tVector.getValue(i));
				} else {
					rfv.add(fv);
					rt.add(tVector.getValue(i));
				}
			}

			final Instances<T> l = new NestedInstances<T>(lfv, lt, this);
			final Instances<T> r = new NestedInstances<T>(rfv, rt, this);
			return Pair.with(l, r);
		}
	}

	static class NestedFeatureColumn<T> implements Vector<T> {

		private static final long serialVersionUID = 1L;

		private final List<Vector<T>> values;
		private final int offset;

		NestedFeatureColumn(final List<Vector<T>> values, final int offset) {
			this.values = values;
			this.offset = offset;
		}

		@Override
		public int getLength() {
			return values.size();
		}

		@Override
		public T getValue(final int index) {
			return values.get(index).getValue(offset);
		}

		@Override
		public Set<T> getUniqueValues() {
			final Set<T> unique = new HashSet<T>(values.size(), 1);
			for (int i = 0; i < values.size(); i++) {
				unique.add(values.get(i).getValue(offset));
			}

			return unique;
		}

		@Override
		public double doubleValue(int anInd) {
			return get(anInd).doubleValue();
		}

		@Override
		public int size() {
			return values.size();
		}

		@Override
		public Number get(int anInd) {
			T value = getValue(anInd);
			if (!(value instanceof Number)) {
				throw new UnsupportedOperationException();
			}

			return (Number) value;
		}

	}

	public static class Builder<T> {

		private final List<T[]> featureVectors;
		private final List<T> targetValues;

		private final SortedMap<Integer, String> featureLabels;
		private String targetLabel;

		private int numCols;

		public Builder() {
			featureVectors = new ArrayList<T[]>(1024);
			targetValues = new ArrayList<T>(1024);

			featureLabels = new TreeMap<Integer, String>();
			targetLabel = "";

			numCols = -1;
		}

		public Builder<T> addInstance(final T targetValue, final T... featureVector) {
			if (targetValue == null || featureVector == null) {
				throw new NullPointerException();
			}

			// TODO: check for nulls in fv

			if (numCols == -1) {
				numCols = featureVector.length;
			} else if (numCols != featureVector.length) {
				throw new IllegalArgumentException();
			}

			featureVectors.add(featureVector);
			targetValues.add(targetValue);
			return this;
		}

		public Builder<T> addInstance(final T targetValue, final List<T> featureVector) {
			if (featureVector == null || targetValue == null) {
				throw new NullPointerException();
			}

			@SuppressWarnings("unchecked")
			T[] arr = featureVector.toArray((T[]) new Object[featureVector.size()]);
			return addInstance(targetValue, arr);
		}

		public Builder<T> setFeatureLabels(List<? extends CharSequence> featureLabels) {
			if (featureLabels != null) {
				for (int i = 0; i < featureLabels.size(); i++) {
					setFeatureLabel(featureLabels.get(i), i);
				}
			}

			return this;
		}

		public Builder<T> setFeatureLabels(final CharSequence... featureLabels) {
			if (featureLabels != null) {
				for (int i = 0; i < featureLabels.length; i++) {
					setFeatureLabel(featureLabels[i], i);
				}
			}

			return this;
		}

		public Builder<T> setFeatureLabel(final CharSequence featureLabel, int featureIndex) {
			if (featureIndex < 0) {
				throw new IllegalArgumentException();
			}

			if (featureLabel != null) {
				featureLabels.put(Integer.valueOf(featureIndex), featureLabel.toString());
			}

			return this;
		}

		public Builder<T> setTargetLabel(final CharSequence targetLabel) {
			if (targetLabel != null) {
				this.targetLabel = targetLabel.toString();
			}

			return this;
		}

		public void validate() {
			// NO-OP
		}

		String[] getFeatureLabels() {
			String[] labels;
			if (numCols > 0) {
				labels = new String[numCols];
				for (int i = 0; i < numCols; i++) {
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

		public InstancesImpl<T> build() {
			validate();
			T[] featureValues = unroll(featureVectors, numCols);
			String[] featureLabels = getFeatureLabels();

			return new InstancesImpl<T>(featureValues, targetValues, featureLabels, targetLabel, featureVectors.size(),
					numCols);
		}

		final static <T> T[] unroll(final List<T[]> segments, final int segmentSize) {
			@SuppressWarnings("unchecked")
			final T[] unrolled = (T[]) new Object[segments.size() * segmentSize];
			for (int i = 0; i < segments.size(); i++) {
				System.arraycopy(segments.get(i), 0, unrolled, i * segmentSize, segmentSize);
			}

			return unrolled;
		}
	}

}
