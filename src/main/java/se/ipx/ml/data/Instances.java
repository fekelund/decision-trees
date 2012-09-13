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
package se.ipx.ml.data;

import java.io.Serializable;

import se.ipx.ml.util.Pair;

/**
 * 
 * @author Fredrik Ekelund
 * 
 * @param <T>
 */
public interface Instances<T> extends Serializable {

	Pair<Instances<T>, Instances<T>> splitUsing(SplitCriteria<T> criteria);

	Matrix<T> getFeatureMatrix();

	Vector<T> getFeatureVector(int index);

	Vector<T> getFeatures(int index);

	Vector<T> getTargets();

	int getNumInstances();

	int getNumFeatures();

	String getTargetLabel();

	String getFeatureLabel(int index);

	String[] getFeatureLabels();

}
