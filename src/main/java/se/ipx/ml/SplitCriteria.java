package se.ipx.ml;

/**
 * 
 * @author Fredrik Ekelund
 * 
 */
public interface SplitCriteria {

	/**
	 * 
	 * @param featureVector
	 * @return
	 */
	boolean isLeft(double[] featureVector);

	/**
	 * 
	 * @param featureVector
	 * @return
	 */
	boolean isRight(double[] featureVector);

}