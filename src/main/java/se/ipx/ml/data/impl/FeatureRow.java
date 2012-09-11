package se.ipx.ml.data.impl;

import java.util.HashSet;
import java.util.Set;

import se.ipx.ml.data.Vector;

class FeatureRow<T> implements Vector<T> {

	private static final long serialVersionUID = 1L;

	private final T[] values;
	private final int offset;
	private final int nBlock;

	FeatureRow(final T[] values, final int offset, final int nBlock) {
		this.values = values;
		this.offset = offset;
		this.nBlock = nBlock;
	}

	@Override
	public int getLength() {
		return nBlock;
	}

	@Override
	public T getValue(final int index) {
		return values[offset + index];
	}

	@Override
	public Set<T> getUniqueValues() {
		final Set<T> unique = new HashSet<T>(nBlock, 1);
		for (int i = 0; i < nBlock; i++) {
			unique.add(values[offset + i]);
		}

		return unique;
	}

	@Override
	public double doubleValue(int anInd) {
		return get(anInd).doubleValue();
	}

	@Override
	public int size() {
		return nBlock;
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