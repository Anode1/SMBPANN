/*
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
package org.smbpann;

import java.util.ArrayList;
import java.util.Iterator;


public class Network {
	
	private ArrayList input;
	private ArrayList output;
	private Hints hints=new Hints(); //way to pass heuristics to the network - when it is necessary (the proper function will get it when it needs, depending on the particular function)
	
	
	//structure
	private ArrayList<Layer> layers; //from input to output
	
	private double step;
	public double currentError; //overall error
	public double goalError; //final error goal

	
	public String name; //name of the network for identification (when running multiple) 
	
	public Network() throws Exception{
		step=hints.getAsDouble(Hints.STEP);
	}
	
	
	/**
	 * One iteration: feed-forward and back propagation 
	 */
	public void learnOnce(TestingSet testingSet) throws Exception{
		
		if(Main.trace) System.out.println(testingSet);
		
		//We build the network from the end (output) to the start (input): the number of outputs defines number of neurons
		//in the last (at least one) layer, as in perceptron. Depending on the number of hidden layers - we connect neurons
		//by edges until we connect the Input with all it's inputs to the 1st layer neurons
		
		//Connect output: we should have at least one output - to read the results from, so we always mapping. Number neurons=number of outputs 
		long sizeOfOutput = testingSet.getOutputSize();
		

		//Also, we need Input and it is meant to be wired with Edges (of at least one neuron as in the perceptron case)
		long sizeOfInput = testingSet.getInputSize();
		
		//layers=new ArrayList();
		
		//iterating through the testing set:
    	Iterator<InputOutput> it = testingSet.iterator();
        while (it.hasNext()) {
        	InputOutput test = it.next();
        	
        }
		
		feedForward();
		backPropagate();
	}
	
	
	public void feedForward() throws Exception{
		
	}
	
	
	public void backPropagate() throws Exception{
		
	}
	
	
	public void setStep(double step){
		this.step=step;
	}
	
	
	public double getCurrentError(){
		return currentError;
	}
	
	
	public Hints getHints(){
		return hints;
	}

}
