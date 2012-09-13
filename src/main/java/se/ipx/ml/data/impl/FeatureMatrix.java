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

import se.ipx.ml.data.Matrix;

class FeatureMatrix<T> implements Matrix<T> {

	private static final long serialVersionUID = 1L;

	private final T[] values;
	private final int offset;
	private final int numRow;
	private final int numCol;
	
	FeatureMatrix(final T[] values, final int offset, final int numRow, final int numCol) {
		this.values = values;
		this.offset = offset;
		this.numRow = numRow;
		this.numCol = numCol;
	}
	
	@Override
	public T getValue(final int row, final int col) {
		return values[offset * row + col];
	}

	@Override
	public int getNumRows() {
		return numRow;
	}

	@Override
	public int getNumCols() {
		return numCol;
	}

	@Override
	public double doubleValue(int aRow, int aCol) {
		return get(aRow, aCol).doubleValue();
	}

	@Override
	public int getColDim() {
		return numCol;
	}

	@Override
	public int getRowDim() {
		return numRow;
	}

	@Override
	public int size() {
		return numRow * numCol;
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