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

public class Network {
	
	private ArrayList input;
	private ArrayList output;
	
	//structure
	private ArrayList<Layer> layers; //from input to output
	
	private double step;
	public double currentError; //overall error
	public double goalError; //final error goal

	
	public String name; //name of the network for identification (when running multiple) 
	
	/**
	 * online (dynamic) learning constructor
	 */
	public Network() throws Exception{
		layers=new ArrayList();
		throw new Exception("Not implmeneted yet!");
	}
	
	
	/**
	 * constructor for batch learning against predetermined inputs/outputs 
	 */
	public Network(ArrayList<TestingSet> testingSet) throws Exception{
		
		step=Parameters.getAsDouble(Constants.STEP_PARAMETER_KEY);
		System.out.println(step);		
		
		layers=new ArrayList();
		
		//construct initial structure here due to some heuristics
		
	}
	
	
	/**
	 * One iteration: feed-forward and back propagation 
	 */
	public void fire() throws Exception{
		//implement me! (iterations through the list of Neurons)
		
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
	
	

}
