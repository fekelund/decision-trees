package se.ipx.ml.data;

import java.io.Serializable;
import java.util.Set;

import org.ojalgo.access.Access1D;

/**
 * 
 * @author Fredrik Ekelund
 * 
 * @param <T>
 */
public interface Vector<T> extends Access1D<Number>, Serializable {

	int getLength();

	T getValue(int index);

	Set<T> getUniqueValues();

}
