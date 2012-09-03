package se.ipx.ml.util;

import java.util.List;

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

	/**
	 * 
	 * @param distribution
	 * @return
	 */
	public static final double mean(final double[] distribution) {
		return sum(distribution) / distribution.length;
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

	/**
	 * Not bias corrected.
	 * 
	 * @param x
	 * @param y
	 * @return
	 */
	public static final double covariance(final double[] x, final double[] y) {
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
	public static final double pearsonsCorrelation(final double[] x, final double[] y) {
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

	/**
	 * 
	 * @param x
	 * @return
	 */
	public static final double[][] transpose(final double[][] x) {
		final int m = x.length;
		final int n = m > 0 ? x[0].length : 0;
		final double[][] t = new double[n][m];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				t[j][i] = x[i][j];
			}
		}

		return t;
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

}
