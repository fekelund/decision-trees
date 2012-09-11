package se.ipx.ml.data;

/**
 * 
 * @author Fredrik Ekelund
 * 
 * @param <T>
 */
public interface SplitCriteria<T> {

	boolean isLeft(Vector<T> vector);

	boolean isRight(Vector<T> vector);

}