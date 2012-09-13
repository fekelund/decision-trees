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

import se.ipx.ml.data.SplitCriteria;
import se.ipx.ml.data.Vector;

class Criteria implements SplitCriteria<Double> {

	private final int feature;
	private final double value;

	private Criteria(final int feature, final double value) {
		this.feature = feature;
		this.value = value;
	}

	static Criteria basedOn(final int feature, final double value) {
		return new Criteria(feature, value);
	}

	@Override
	public boolean isLeft(final Vector<Double> featureVector) {
		return featureVector.getValue(feature).doubleValue() >= value;
	}

	@Override
	public boolean isRight(final Vector<Double> featureVector) {
		return featureVector.getValue(feature).doubleValue() < value;
	}

}