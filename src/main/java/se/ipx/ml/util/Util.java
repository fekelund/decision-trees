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
package se.ipx.ml.util;

import java.util.Iterator;
import java.util.List;

import se.ipx.ml.data.Vector;

public class Util {

	private Util() {
	}

	/**
	 * 
	 * @param distribution
	 * @return
	 */
	public static final double sum(final double[] distribution) {
		double sum = 0D;
		for (double value : distribution) {
			sum += value;
		}

		return sum;
	}

	public static final double sum(final Iterable<Double> distribution) {
		double sum = 0D;
		for (Double value : distribution) {
			sum += value;
		}
		
		return sum;
	}

	public static final double sum(final Vector<Double> vector) {
		double sum = 0D;
		for (int i = 0; i < vector.getLength(); i++) {
			sum += vector.getValue(i);
		}
		
		return sum;
	}

	/**
	 * 
	 * @param distribution
	 * @return
	 */
	public static final double mean(final double[] distribution) {
		return sum(distribution) / distribution.length;
	}

	public static final double mean(final Iterable<Double> distribution, final int length) {
		return sum(distribution) / length;
	}

	public static final double mean(final Vector<Double> vector) {
		return sum(vector) / vector.getLength();
	}
	
	/**
	 * Not bias corrected.
	 * 
	 * @param distribution
	 * @return
	 */
	public static final double variance(final double[] distribution) {
		final double mean = mean(distribution);
		double sum1 = 0D, sum2 = 0D, deviation = 0D;
		for (int i = 0; i < distribution.length; i++) {
			deviation = distribution[i] - mean;
			sum1 += deviation * deviation;
			sum2 += deviation;
		}

		return (sum1 - (sum2 * sum2 / distribution.length)) / distribution.length;
	}

	public static final double variance(final Iterable<Double> distribution, final int length) {
		final double mean = mean(distribution, length);
		double sum1 = 0D, sum2 = 0D, deviation = 0D;
		for (Double value : distribution) {
			deviation = value - mean;
			sum1 += deviation * deviation;
			sum2 += deviation;
		}
		
		return (sum1 - (sum2 * sum2 / length)) / length;
	}

	public static final double variance(final Vector<Double> vector) {
		final double mean = mean(vector);
		double sum1 = 0D, sum2 = 0D, deviation = 0D;
		for (int i = 0; i < vector.getLength(); i++) {
			deviation = vector.getValue(i) - mean;
			sum1 += deviation * deviation;
			sum2 += deviation;
		}

		return (sum1 - (sum2 * sum2 / vector.getLength())) / vector.getLength();
	}
	
	/**
	 * Not bias corrected.
	 * 
	 * @param x
	 * @param y
	 * @return
	 */
	public static final double covariance(final double[] x, final double[] y) {
		if (x.length != y.length) {
			throw new IllegalArgumentException();
		}
		
		final double prodXY = sum(x) * sum(y);
		double sum = 0D;
		double covariance = 0D;
		for (int i = 0; i < x.length; i++) {
			sum += x[i] * y[i];
			covariance = (sum - prodXY / x.length) / x.length;
		}

		return covariance;
	}

	/**
	 * Not bias corrected.
	 * 
	 * @param distribution
	 * @return
	 */
	public static double standardDeviation(final double[] distribution) {
		return Math.sqrt(variance(distribution));
	}

	/**
	 * 
	 * @param x
	 * @param y
	 * @return
	 */
	public static final double correlation(final double[] x, final double[] y) {
		final double meanX = mean(x);
		final double meanY = mean(y);
		double sumXY = 0.0, sumX2 = 0.0, sumY2 = 0.0;
		for (int i = 0; i < x.length; i++) {
			final double dx = x[i] - meanX;
			final double dy = y[i] - meanY;
			sumXY += dx * dy;
			sumX2 += dx * dx;
			sumY2 += dy * dy;
		}

		return (sumXY / (Math.sqrt(sumX2) * Math.sqrt(sumY2)));
	}

	public static final double correlation(final Vector<Double> x, final Vector<Double> y) {
		final double meanX = mean(x);
		final double meanY = mean(y);
		double sumXY = 0.0, sumX2 = 0.0, sumY2 = 0.0;
		for (int i = 0; i < x.getLength(); i++) {
			final double dx = x.getValue(i) - meanX;
			final double dy = y.getValue(i) - meanY;
			sumXY += dx * dy;
			sumX2 += dx * dx;
			sumY2 += dy * dy;
		}
		
		return (sumXY / (Math.sqrt(sumX2) * Math.sqrt(sumY2)));
	}

	public static final double correlation(final double[] x, final Vector<Double> y) {
		final double meanX = mean(x);
		final double meanY = mean(y);
		double sumXY = 0.0, sumX2 = 0.0, sumY2 = 0.0;
		for (int i = 0; i < x.length; i++) {
			final double dx = x[i] - meanX;
			final double dy = y.getValue(i) - meanY;
			sumXY += dx * dy;
			sumX2 += dx * dx;
			sumY2 += dy * dy;
		}
		
		return (sumXY / (Math.sqrt(sumX2) * Math.sqrt(sumY2)));
	}
	
	public static final double correlation(final Iterable<Double> x, final Iterable<Double> y, int length) {
		final double meanX = mean(x, length);
		final double meanY = mean(y, length);
		double sumXY = 0.0, sumX2 = 0.0, sumY2 = 0.0;
		Iterator<Double> xi = x.iterator();
		Iterator<Double> yi = y.iterator();
		while (xi.hasNext() && yi.hasNext()) {
			final double dx = xi.next() - meanX;
			final double dy = yi.next() - meanY;
			sumXY += dx * dy;
			sumX2 += dx * dx;
			sumY2 += dy * dy;
		}
		
		return (sumXY / (Math.sqrt(sumX2) * Math.sqrt(sumY2)));
	}

	/**
	 * 
	 * @param A
	 * @return
	 */
	public static final double[][] transpose(final double[][] A) {
		final int m = A.length;
		final int n = m > 0 ? A[0].length : 0;
		final double[][] aT = new double[n][m];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				aT[j][i] = A[i][j];
			}
		}

		return aT;
	}

	/**
	 * 
	 * @param a
	 * @return
	 */
	public static final double[][] transpose(final double[] a) {
		final int m = a.length;
		final double[][] aT = new double[m][1];
		for (int i = 0; i < m; i++) {
			aT[i][0] = a[i];
		}
		
		return aT;
	}

	/**
	 * 
	 * @param numbers
	 * @return
	 */
	public static final double[] convert(final Number... numbers) {
		final double[] primitives = new double[numbers.length];
		for (int i = 0; i < numbers.length; i++) {
			primitives[i] = numbers[i].doubleValue();
		}

		return primitives;
	}

	public static final double[] convert(final Vector<? extends Number> numbers) {
		final double[] primitives = new double[numbers.getLength()];
		for (int i = 0; i < primitives.length; i++) {
			primitives[i] = numbers.getValue(i).doubleValue();
		}
		
		return primitives;
	}

	/**
	 * 
	 * @param numbers
	 * @return
	 */
	public static final double[] convert(final List<? extends Number> numbers) {
		final double[] primitives = new double[numbers.size()];
		for (int i = 0; i < numbers.size(); i++) {
			primitives[i] = numbers.get(i).doubleValue();
		}

		return primitives;
	}

	/**
	 * 
	 * @param scalar
	 * @return
	 */
	public static final double[][] asMatrix(final double scalar) {
		return new double[][] {{ scalar }};
	}

	/**
	 * 
	 * @param vector
	 * @return
	 */
	public static final double[][] asMatrix(final double[] vector) {
		if (vector.length == 0) {
			return new double[0][0];						
		} else {
			return new double[][] { vector };			
		}
	}

	public static final double asScalar(final double[][] matrix) {
		if (matrix.length != 1 || matrix[0].length != 1) {
			throw new IllegalArgumentException();
		}
		
		return matrix[0][0];
	}
	
	/**
	 * 
	 * @param A
	 * @param B
	 * @return
	 */
	public static final double[][] multiply(final double[][] A, final double[][] B) {
		final int n = A.length;
		final int m = B.length;
		if (m == 0 || n == 0) {
			return new double[0][0];
		}
		
		final int p = B[0].length;
		if (m != A[0].length) {
			throw new IllegalArgumentException(String.format("Non-conformant matrix dimensions: [%d x %d] * [%d x %d]", n, A[0].length, m, p));
		}
		
		if (p == 0) {
			return new double[0][0];
		}
		
//		return uncheckedMultiply(A, B, n, m, p);
		return uncheckedMultiplyJAMA(A, B, n, m, p);
	}
	
	/**
	 * 
	 * @param A
	 * @param B
	 * @return
	 */
	public static final double[][] uncheckedMultiply(final double[][] A, final double[][] B) {
//		return uncheckedMultiply(A, B, A.length, B.length, B[0].length);
		return uncheckedMultiplyJAMA(A, B, A.length, B.length, B[0].length);
	}
	
	static final double[][] uncheckedMultiply(final double[][] A, final double[][] B, final int n, final int m, final int p) {
		final double[][] C = new double[n][p];
		for (int i = 0; i < n; i++) {
			final double[] aRow = A[i];
			final double[] cRow = C[i];
			for (int k = 0; k < m; k++) {
				final double[] bRow = B[k];
				final double aVal = aRow[k];
				for (int j = 0; j < p; j++) {
					cRow[j] += aVal * bRow[j];
				}
			}
		}
		
		return C;
	}
	
	static double[][] uncheckedMultiplyJAMA(final double[][] A, final double[][] B, final int n, final int m, final int p) {
		final double[][] C = new double[n][p];
		final double[] bCol = new double[m];
		for (int j = 0; j < p; j++) {
			for (int k = 0; k < m; k++)
				bCol[k] = B[k][j];
			for (int i = 0; i < n; i++) {
				final double[] aRow = A[i];
				double sum = 0.0;
				for (int k = 0; k < m; k++) {
					sum += aRow[k] * bCol[k];
				}
				
				C[i][j] = sum;
			}
		}
		
		return C;
	}
}
