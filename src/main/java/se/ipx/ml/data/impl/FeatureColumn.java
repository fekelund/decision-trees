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

import java.util.HashSet;
import java.util.Set;

import se.ipx.ml.data.Vector;

class FeatureColumn<T> implements Vector<T> {

	private static final long serialVersionUID = 1L;

	private final T[] values;
	private final int offset;
	private final int nBlock;
	private final int length;

	FeatureColumn(final T[] values, final int offset, final int nBlock) {
		this.values = values;
		this.offset = offset;
		this.nBlock = nBlock;
		this.length = values.length / nBlock;
	}

	@Override
	public int getLength() {
		return length;
	}

	@Override
	public T getValue(final int index) {
		return values[(index * nBlock) + offset];
	}

	@Override
	public Set<T> getUniqueValues() {
		final Set<T> unique = new HashSet<T>(length, 1);
		for (int i = 0; i < length; i++) {
			unique.add(values[(i * nBlock) + offset]);
		}

		return unique;
	}

	@Override
	public double doubleValue(int anInd) {
		return get(anInd).doubleValue();
	}

	@Override
	public int size() {
		return length;
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