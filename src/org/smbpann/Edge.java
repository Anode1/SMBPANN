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

/**
 * Always points to some Node (Neurons), this is like a pointer 
 */
public class Edge {

	private Neuron leftNeuron; //The Neuron at the left
	private Neuron rightNeuron; //The Neuron at the right
	
	public double weight;
	public double weightDiff; //last weight difference
	public double error; //current error
	public double de; //error difference
	

	public Edge(Neuron leftNeuron, Neuron rightNeuron){
		
		this.leftNeuron=leftNeuron;
		leftNeuron.addOutgoingEdge(this);
		
		this.rightNeuron=rightNeuron;
		rightNeuron.addIncomingEdge(this);
		
		weight=Math.random(); //initialize by random (0..1) for now (we might want to have some flexibility here in future)
	}	

	/*
	public double getLeftData(){
		//leftNeuron
	}*/
	
	
	/**
	 * Get the Neuron at the left 
	 */
	public Neuron getLeftNeuron(){
		return leftNeuron;
	}
	
	
	/**
	 * Get the Neuron at the right 
	 */
	public Neuron getRightNeuron(){
		return rightNeuron;
	}
	
	
	/**
	 * For Id purposes [leftNodeID,rightNodeID]
	 */
	public String getName(){
		return "["+leftNeuron.getName()+"-"+rightNeuron.getName()+"]";
	}
	
	
	/**
	 * For debugging purposes only
	 */
	public String toString(){
		return "edge:"+getName()+", weight:"+weight+", error: "+error+", de:"+de;
	}
	
}
