package org.smbpann;


/**
 * Non-linear activation functions (we likely need just one)
 */
public class Activation {
	
	/**
	 * Currently sigmoid, but dynamic change of function may be implemented here
	 */
	public static double activate(double x){
		return sigmoid(x);
	}

	
	// We should probably store cached tables (has to be measured yet)	
	public static double sigmoid(double x){
		//slope parameter not shown 
		return 1/(1 + Math.exp(-x));
	}
	
	
	// We should probably store cached tables (has to be measured yet)
	public static double tanSigmoid(double x){
		return 1 - Math.pow((Math.exp(x)-Math.exp(-x)), 2)/Math.pow((Math.exp(x)+Math.exp(-x)), 2);
	}
	
	
	public static double softmax(double[] x, int n){
		double sum=0;
		for(int i=0; i<x.length; i++){
			sum += Math.exp(x[i]);
		}
		return Math.exp(x[n])/sum;
	}
	
}
