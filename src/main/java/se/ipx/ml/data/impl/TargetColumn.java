package se.ipx.ml.data.impl;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import se.ipx.ml.data.Vector;

class TargetColumn<T> implements Vector<T> {

	private static final long serialVersionUID = 1L;

	private final List<T> values;

	TargetColumn(final List<T> values) {
		this.values = values;
	}

	@Override
	public int getLength() {
		return values.size();
	}

	@Override
	public T getValue(final int index) {
		return values.get(index);
	}

	@Override
	public Set<T> getUniqueValues() {
		return new HashSet<T>(values);
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