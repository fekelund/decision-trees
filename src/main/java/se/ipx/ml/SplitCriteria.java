package se.ipx.ml;

public interface SplitCriteria {
	
	boolean isLeft(double[] featureVector);
	
	boolean isRight(double[] featureVector);
	
}