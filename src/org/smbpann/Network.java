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
	private ArrayList layers;
	private double currentError;
	private double goalError;
	private boolean terminating;

	
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
	public Network(ArrayList input, ArrayList output) throws Exception{
		layers=new ArrayList();
	}
	
	
	public void process() throws Exception{
		try{
			while(!terminating){
				
				//implement me! (iterations through the list of Neurons)
				
				feedForward();
				backPropagate();
			}
		}
		finally{
			if(Main.trace) System.out.println("stopped");
		}
	}
	
	
	public void feedForward() throws Exception{
		
	}
	
	
	public void backPropagate() throws Exception{
		
	}
	
	
	public double getCurrentError(){
		return currentError;
	}
	
	
	public void stop(){
		terminating=true;
	}
	
}
