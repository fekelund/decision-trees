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

import java.math.BigDecimal;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.testng.Assert;
import org.testng.annotations.Test;

public class UtilTest {

	public final static double[][] M_1x1 = {{ 2 }};
	public final static double[]   V_1x1 = {  2  };
	
	public final static double[][] M_3x1 = {{ 1 }, 
											{ 2 }, 
											{ 3 }};
	
	public final static double[][] M_1x3 = {{ 1, 2, 3 }};
	public final static double[]   V_1x3 = {  1, 2, 3  };
	
	public final static double[][] M_3x3 = {{ 1, 2, 3 }, 
											{ 4, 5, 6 }, 
											{ 7, 8, 9 }};
	
	public final static double[][] EM_1 = new double[0][0];
	public final static double[][] EM_2 = new double[0][1];
	public final static double[][] EM_3 = new double[1][0];
	public final static double[]   EV_1 = new double[0];
	

	
	@Test
	public void testMatrixMultiplication_1x1_x_1x1() {
		// [ 4 ]
		double[][] p = Util.multiply(M_1x1, M_1x1);
		Assert.assertEquals(4D, Util.asScalar(p));
	}

	@Test
	public void testMatrixMultiplication_1x1_x_1x3() {
		// [ 2 4 6 ]
		double[][] p = Util.multiply(M_1x1, M_1x3);
		Assert.assertEquals(p.length, 1);
		Assert.assertEquals(p[0].length, 3);
		Assert.assertEquals(p[0][0], 2D);
		Assert.assertEquals(p[0][1], 4D);
		Assert.assertEquals(p[0][2], 6D);
	}
	
	@Test
	public void testMatrixMultiplication_3x1_x_1x1() {
		// [ 1 ] 
		// | 4 |
		// [ 6 ]
		double[][] p = Util.multiply(M_3x1, M_1x1);
		Assert.assertEquals(p.length, 3);
		Assert.assertEquals(p[0].length, 1);
		Assert.assertEquals(p[0][0], 2D);
		Assert.assertEquals(p[1][0], 4D);
		Assert.assertEquals(p[2][0], 6D);
	}
	
	@Test
	public void testMatrixMultiplication_3x1_x_1x3() {
		// [ 1 2 3 ]
		// | 2 4 6 |
		// [ 3 6 9 ] 
		double[][] p = Util.multiply(M_3x1, M_1x3);
		Assert.assertEquals(p.length, 3);
		Assert.assertEquals(p[0].length, 3);
		Assert.assertEquals(p[0][0], 1D);
		Assert.assertEquals(p[0][1], 2D);
		Assert.assertEquals(p[0][2], 3D);
		Assert.assertEquals(p[1][0], 2D);
		Assert.assertEquals(p[1][1], 4D);
		Assert.assertEquals(p[1][2], 6D);
		Assert.assertEquals(p[2][0], 3D);
		Assert.assertEquals(p[2][1], 6D);
		Assert.assertEquals(p[2][2], 9D);
	}

	@Test
	public void testMatrixMultiplication_1x3_x_3x1() {
		// [ 14 ]
		double[][] p = Util.multiply(M_1x3, M_3x1);
		Assert.assertEquals(p.length, 1);
		Assert.assertEquals(p[0].length, 1);
		Assert.assertEquals(p[0][0], 14D);
	}
	
	@Test
	public void testMatrixMultiplication_3x3_x_3x1() {
		// [ 14 ] 
		// | 32 |
		// [ 50 ]
		double[][] p = Util.multiply(M_3x3, M_3x1);
		Assert.assertEquals(p.length, 3);
		Assert.assertEquals(p[0].length, 1);
		Assert.assertEquals(p[0][0], 14D);
		Assert.assertEquals(p[1][0], 32D);
		Assert.assertEquals(p[2][0], 50D);
	}
	
	@Test
	public void testMatrixMultiplication_1x3_x_3x3() {
		// [ 30 36 42 ]
		double[][] p = Util.multiply(M_1x3, M_3x3);
		Assert.assertEquals(p.length, 1);
		Assert.assertEquals(p[0].length, 3);
		Assert.assertEquals(p[0][0], 30D);
		Assert.assertEquals(p[0][1], 36D);
		Assert.assertEquals(p[0][2], 42D);
	}
	
	@Test
	public void testMatrixMultiplication_3x3_x_3x3() {
		// [  30  36  42 ]
		// |  66  81  96 |
		// [ 102 126 150 ] 
		double[][] p = Util.multiply(M_3x3, M_3x3);
		Assert.assertEquals(p.length, 3);
		Assert.assertEquals(p[0].length, 3);
		Assert.assertEquals(p[0][0], 30D);
		Assert.assertEquals(p[0][1], 36D);
		Assert.assertEquals(p[0][2], 42D);
		Assert.assertEquals(p[1][0], 66D);
		Assert.assertEquals(p[1][1], 81D);
		Assert.assertEquals(p[1][2], 96D);
		Assert.assertEquals(p[2][0], 102D);
		Assert.assertEquals(p[2][1], 126D);
		Assert.assertEquals(p[2][2], 150D);
	}
	
	@Test
	public void testMatrixMultiplcation_empty() {
		double[][] p = Util.multiply(EM_1, EM_1);
		Assert.assertEquals(p.length, 0);
		p = Util.multiply(EM_1, EM_2);
		Assert.assertEquals(p.length, 0);
		p = Util.multiply(EM_2, EM_3);
		Assert.assertEquals(p.length, 0);
		p = Util.multiply(EM_3, EM_1);
		Assert.assertEquals(p.length, 0);
	}
	
	@Test(expectedExceptions=IllegalArgumentException.class)
	public void testMatrixMultiplcation_3x1_x_3x1() {
		Util.multiply(M_3x1, M_3x1);
	}

	@Test(expectedExceptions=IllegalArgumentException.class)
	public void testMatrixMultiplcation_1x3_x_1x3() {
		Util.multiply(M_1x3, M_1x3);
	}

	@Test(expectedExceptions=IllegalArgumentException.class)
	public void testMatrixMultiplcation_3x3_x_1x3() {
		Util.multiply(M_3x3, M_1x3);
	}
	
	@Test(expectedExceptions=IllegalArgumentException.class)
	public void testMatrixMultiplcation_3x1_x_3x3() {
		Util.multiply(M_3x3, M_1x3);
	}
	
	@Test
	public void testMatrixTranspose_1x1() {
		double[][] tM_1x1 = Util.transpose(M_1x1);
		Assert.assertEquals(tM_1x1.length, 1);
		Assert.assertEquals(tM_1x1[0].length, 1);
		Assert.assertEquals(tM_1x1[0][0], 2D);
	}
	
	@Test
	public void testMatrixTranspose_3x3() {
		double[][] tM_3x3 = Util.transpose(M_3x3);
		Assert.assertEquals(tM_3x3.length, 3);
		Assert.assertEquals(tM_3x3[0].length, 3);
		Assert.assertEquals(tM_3x3[0][0], 1D);
		Assert.assertEquals(tM_3x3[0][1], 4D);
		Assert.assertEquals(tM_3x3[0][2], 7D);
		Assert.assertEquals(tM_3x3[1][0], 2D);
		Assert.assertEquals(tM_3x3[1][1], 5D);
		Assert.assertEquals(tM_3x3[1][2], 8D);
		Assert.assertEquals(tM_3x3[2][0], 3D);
		Assert.assertEquals(tM_3x3[2][1], 6D);
		Assert.assertEquals(tM_3x3[2][2], 9D);
	}
	
	@Test
	public void testVectorTranspose_1x3() {
		double[][] tV_3x1 = Util.transpose(V_1x3);
		Assert.assertEquals(tV_3x1.length, 3);
		Assert.assertEquals(tV_3x1[0].length, 1);
		Assert.assertEquals(tV_3x1[0][0], 1D);
		Assert.assertEquals(tV_3x1[1][0], 2D);
		Assert.assertEquals(tV_3x1[2][0], 3D);
	}

	@Test
	public void testVectorTranspose_3x1() {
		double[][] tM_1x3 = Util.transpose(M_3x1);
		Assert.assertEquals(tM_1x3.length, 1);
		Assert.assertEquals(tM_1x3[0].length, 3);
		Assert.assertEquals(tM_1x3[0][0], 1D);
		Assert.assertEquals(tM_1x3[0][1], 2D);
		Assert.assertEquals(tM_1x3[0][2], 3D);
	}

	@Test
	public void testVectorTranspose_1x1() {
		double[][] tV_1x1 = Util.transpose(V_1x1);
		Assert.assertEquals(tV_1x1.length, 1);
		Assert.assertEquals(tV_1x1[0].length, 1);
		Assert.assertEquals(tV_1x1[0][0], 2D);
	}

	@Test
	public void testMatrixTranspose_empty() {
		double[][] tE_1 = Util.transpose(EM_1);
		Assert.assertEquals(tE_1.length, 0);
		double[][] tE_2 = Util.transpose(EM_2);
		Assert.assertEquals(tE_2.length, 0);
		double[][] tE_3 = Util.transpose(EM_3);
		Assert.assertEquals(tE_3.length, 0);
	}
	
	@Test
	public void testVectorTranspose_empty() {
		double[][] tEV_1 = Util.transpose(EV_1);
		Assert.assertEquals(tEV_1.length, 0);
	}
	
	@Test
	public void testSum() {
		double sum = Util.sum(V_1x3);
		Assert.assertEquals(sum, 6D);
	}

	@Test
	public void testSum_empty() {
		double sum = Util.sum(EV_1);
		Assert.assertEquals(sum, 0D);
	}
	
	@Test
	public void testMean_1() {
		Assert.assertEquals(Util.mean(new double[] {  1, 2, 3 }), 2D);
	}

	@Test
	public void testMean_2() {
		Assert.assertEquals(Util.mean(new double[] {  4, 5, 6, 7 }), 5.5D);
	}

	@Test
	public void testMean_3() {
		Assert.assertEquals(Util.mean(new double[] {  4, 5, 6, 6, 7, 8 }), 6D);
	}

	@Test
	public void testMean_single() {
		Assert.assertEquals(Util.mean(new double[] { 1 }), 1D);
	}

	@Test
	public void testMean_empty() {
		Assert.assertEquals(Util.mean(EV_1), Double.NaN);
	}

	@Test
	public void testVariance_1() {
		Assert.assertEquals(Util.variance(new double[] { 1, 2, 3 }), 0.6666666666666666D);
	}
	
	@Test
	public void testVariance_2() {
		Assert.assertEquals(Util.variance(new double[] { 1, 1, 2 }), 0.2222222222222222D);
	}
	
	@Test
	public void testVariance_3() {
		Assert.assertEquals(Util.variance(new double[] { 1, 1, 1 }), 0D);
	}

	@Test
	public void testVariance_empty() {
		Assert.assertEquals(Util.variance(new double[] { }), Double.NaN);
	}
	
//	@Test
//	public void testCovariance_1() {
//		double[] x = new double[] { 1, 1, 1 };
//		double[] y = new double[] { 1, 1, 1 };
//		Assert.assertEquals(Util.covariance(x, y), 0D);
//	}
//	
//	@Test
//	public void testCovariance_2() {
//		double[] x = new double[] { 1, 2, 3 };
//		double[] y = new double[] { 1, 2, 3 };
//		Assert.assertEquals(Util.covariance(x, y), 1D); //0.6666666666666666
//	}
//	
//	@Test
//	public void testCovariance_3() {
//		double[] x = new double[] { 1, 2, 3 };
//		double[] y = new double[] { 3, 2, 1 };
//		Assert.assertEquals(Util.covariance(x, y), -1D);//-0.6666666666666666
//	}
//	
//	@Test
//	public void testCovariance_4() {
//		double[] x = new double[] { 1, 2, 3 };
//		double[] y = new double[] { 1, 2, 4 };
//		Assert.assertEquals(Util.covariance(x, y), 1.5D);//1
//	}
//	
//	@Test
//	public void testCovariance_5() {
//		double[] x = new double[] { 1, 2, 3 };
//		double[] y = new double[] { 1, 0, 0 };
//		Assert.assertEquals(Util.covariance(x, y), -0.5D);//-0.3333333333333333
//	}

	@Test(expectedExceptions=IllegalArgumentException.class)
	public void testCovariance_dimensionDiff() {
		double[] x = new double[] { 1, 2 };
		double[] y = new double[] { 1, 2, 3 };
		Util.covariance(x, y);
	}

	@Test
	public void testCovariance_empty() {
		double[] x = new double[] { };
		double[] y = new double[] { };
		Assert.assertEquals(Util.covariance(x, y), 0D);
	}
	
	@Test
	public void testStandardDeviation_() {
		// TODO: implement
	}

	@Test
	public void testPearsonsCorrelation_() {
		// TODO: implement
	}

	@Test
	public void testConvertVararg_1() {
		double[] c = Util.convert(1, 2, 3);
		Assert.assertEquals(c.length, 3);
		Assert.assertEquals(c[0], 1D);
		Assert.assertEquals(c[1], 2D);
		Assert.assertEquals(c[2], 3D);
	}

	@Test
	public void testConvertVararg_2() {
		double[] c = Util.convert(BigDecimal.valueOf(1), BigDecimal.valueOf(2), BigDecimal.valueOf(3));
		Assert.assertEquals(c.length, 3);
		Assert.assertEquals(c[0], 1D);
		Assert.assertEquals(c[1], 2D);
		Assert.assertEquals(c[2], 3D);
	}

	@Test
	public void testConvertVararg_empty() {
		double[] c = Util.convert();
		Assert.assertEquals(c.length, 0);
	}

	@Test
	public void testConvertList_1() {
		double[] c = Util.convert(Arrays.asList(1, 2, 3));
		Assert.assertEquals(c.length, 3);
		Assert.assertEquals(c[0], 1D);
		Assert.assertEquals(c[1], 2D);
		Assert.assertEquals(c[2], 3D);
	}

	@Test
	public void testConvertList_empty() {
		List<Number> l = Collections.emptyList();
		double[] c = Util.convert(l);
		Assert.assertEquals(c.length, 0);
	}

	@Test
	public void asMatrix() {
		double[][] m = Util.asMatrix(2D);
		Assert.assertEquals(m.length, 1);
		Assert.assertEquals(m[0].length, 1);
		Assert.assertEquals(m[0][0], 2D);
	}

	@Test
	public void asMatrix_1x1() {
		double[][] m = Util.asMatrix(V_1x1);
		Assert.assertEquals(m.length, 1);
		Assert.assertEquals(m[0].length, 1);
		Assert.assertEquals(m[0][0], 2D);
	}

	@Test
	public void asMatrix_1x3() {
		double[][] m = Util.asMatrix(V_1x3);
		Assert.assertEquals(m.length, 1);
		Assert.assertEquals(m[0].length, 3);
		Assert.assertEquals(m[0][0], 1D);
		Assert.assertEquals(m[0][1], 2D);
		Assert.assertEquals(m[0][2], 3D);
	}

	@Test
	public void asMatrix_empty() {
		double[][] m = Util.asMatrix(EV_1);
		Assert.assertEquals(m.length, 0);
	}

	@Test
	public void asScalar_1x1() {
		double s = Util.asScalar(M_1x1);
		Assert.assertEquals(s, 2D);
	}

	@Test(expectedExceptions = IllegalArgumentException.class)
	public void asScalar_1x3() {
		Util.asScalar(M_1x3);
	}

	@Test(expectedExceptions = IllegalArgumentException.class)
	public void asScalar_3x1() {
		Util.asScalar(M_3x1);
	}

	public static void main(String[] args) {
	}
}
