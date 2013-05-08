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

import se.ipx.ml.data.Instance;
import se.ipx.ml.data.Instances;
import se.ipx.ml.data.Vector;

/**
 * 
 * @author Fredrik Ekelund
 * 
 * @param <T>
 */
class InstanceImpl<T> implements Instance<T> {

	private final Instances<T> instances;
	private final T target;
	private final int index;

	InstanceImpl(final Instances<T> instances, final T target, final int index) {
		this.instances = instances;
		this.target = target;
		this.index = index;
	}

	@Override
	public Vector<T> getFeatureVector() {
		return instances.getFeatureVector(index);
	}

	@Override
	public T getTargetValue() {
		return target;
	}

}