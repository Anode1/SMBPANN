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

public class Neuron{

	private String name; //name of the neuron for easier identification
	private double value; //current value
	private double desiredValue; //desired value (in output neurons)
	private double bias;
	private ArrayList<Edge> incomingEdges=new ArrayList<Edge>(); //inputs
	private ArrayList<Edge> outgoingEdges=new ArrayList<Edge>(); //outputs (although those are inputs for other nodes - we need ref here for forward feeding
	
	
	public Neuron(){
	}
	
	
	public Neuron(String name){
		this.name=name;
	}
	
	
	public void feedForward() throws Exception{
		ArrayList<Edge> incomingEdges = getIncomingEdges();
		int numberOfInputs=incomingEdges.size();
		double sum=0;
		Iterator<Edge> it = incomingEdges.iterator();
		while(it.hasNext()){
			Edge edge=it.next();
			double input = edge.getLeftNeuron().getOutputValue();
			sum += input * edge.weight;
		}
		value=Activation.activate(sum); //sigmoid currently but we can experiment with different activation functions (tansigmoid, softmax) there 
	}
	
	
	public void backPropagate(boolean isLastLayer) throws Exception{
		
		//error:
		//edge.dx = value * (1 - value) * sum; //derivative of sigmoid
		
		ArrayList<Edge> incomingEdges = getIncomingEdges();
		Iterator<Edge> it = incomingEdges.iterator();
		while(it.hasNext()){
			Edge edge=it.next();
			
			edge.weight+=Network.learningRate * edge.error * edge.getLeftNeuron().getOutputValue();
			
			if(isLastLayer){
				
			}
			else{
				
			}
			
		}
	}
	

	/*
	public void printWithChildrenRecursively(StringBuffer out){ //TODO: change to PrintWriter to be able to dump big trees
    	out.append("neuron:"+getName());
    	out.append("\n");
		Iterator<Edge> it = outgoingEdges.iterator();
        while(it.hasNext()) {
        	Edge edge = it.next();
        	out.append(edge);
        	out.append("\n");
        	Neuron neuron = edge.getNeuron();
        	out.append(neuron);
        	out.append("\n");
        	neuron.printWithChildrenRecursively(out);
        }
	}*/
	
	
	public ArrayList<Edge> getIncomingEdges(){
		return incomingEdges;
	}
	
	
	public void addIncomingEdge(Edge edge){
		incomingEdges.add(edge);
	}
	
	
	public void removeIncomingEdge(Edge edge){
		incomingEdges.remove(edge);
	}
	
	
	public ArrayList<Edge> getOutgoingEdges(){
		return outgoingEdges;
	}
	
	
	public void addOutgoingEdge(Edge edge){
		outgoingEdges.add(edge);
	}
	

	public void removeOutgoingEdge(Edge edge){
		outgoingEdges.remove(edge);
	}	
	
	
	public double getOutputValue(){
		return value;
	}
	
	
	public double getDesiredValue(){
		return value;
	}
	
	
	public void setDesiredValue(double desiredValue){
		this.desiredValue=desiredValue;
	}
	
	
	public String getName(){
		return name;
	}
	
	
	void setName(String name){
		this.name=name;
	}
	
	
	void setValue(double value){
		this.value=value;
	}
	
	
	/**
	 * For debugging purposes only
	 */
	public String toString(){
		StringBuffer sb=new StringBuffer();
		sb.append("neuron:"+name);
        sb.append(", output="+Double.toString(value));
		
    	Iterator<Edge> it = incomingEdges.iterator();
    	sb.append(", incoming: [");
        while (it.hasNext()) {
        	Edge edge = it.next();
        	sb.append(edge.getName());
        	if(it.hasNext())sb.append(",");
        }
        sb.append("]");
        
        it = outgoingEdges.iterator();
        sb.append(", outgoing: [");
        while (it.hasNext()) {
        	Edge edge = it.next();
        	sb.append(edge.getName());
        	if(it.hasNext())sb.append(",");
        }
        sb.append("]");
		return sb.toString();
	}	
	
}
