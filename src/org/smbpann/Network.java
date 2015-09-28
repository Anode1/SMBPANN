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
		
		//we should have at least one output (to read the results from), so we always mapping 
		//at least one neuron in the output layer to the Output

		//Also, we need Input and it is meant to be wired with Edges
		
		//layers=new ArrayList();
		
		//iterating through the testing set:
    	Iterator<TestingExample> it = testingSet.iterator();
        while (it.hasNext()) {
        	TestingExample test = it.next();
        	
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
