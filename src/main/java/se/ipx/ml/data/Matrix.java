package se.ipx.ml.data;

import java.io.Serializable;

import org.ojalgo.access.Access2D;

/**
 * 
 * @author Fredrik Ekelund
 * 
 * @param <T>
 */
public interface Matrix<T> extends Access2D<Number>, Serializable {

	T getValue(int row, int col);

	int getNumRows();

	int getNumCols();

}