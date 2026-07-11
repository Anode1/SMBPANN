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


/**
 * Input/Output terminals of the Network (were the data is written). Since the inputs has fixed dimensions, in contrast with 
 * hidden layers the Input/Output Terminals are not changing. This class keeps references to the static part of the Neural
 * Network which will not be changed (since it is strictly defined by Input/Output sizes). All other Nodes/Neurons may be 
 * changed, added, differently interconnected by Edges
 */
public class InputOutputTerminals {
	
	
	private ArrayList<Neuron> inputNodes = new ArrayList<Neuron>(); //not calculating Neurons (place holders for data)
	private ArrayList<Neuron> outputNeurons = new ArrayList<Neuron>(); //the last layer
	
	
	/**
	 * Create input/output terminals, including dummy (not calculating) Nodes of the input and actual calculating
	 * Neurons of the last layer
	 */
	public InputOutputTerminals(TestingSet testingSet) throws Exception{
		
    	//Create not calculating Neurons (terminal Nodes) for each Input (it will not be included into a Layer)
		int inputSize = testingSet.getInputSize();
		for(int i=0; i<inputSize; i++){
        	Neuron inputNeuron = new Neuron("input "+i);
        	inputNodes.add(inputNeuron);
		}
        	
		//create calculating Neurons for each Output (those will be included into the last Layer) 
		int outputSize = testingSet.getOutputSize();
		for(int i=0; i<outputSize; i++){
			Neuron neuron=new Neuron(Integer.toString(i)); 
			outputNeurons.add(neuron);
		}
	}
	
	
	/**
	 * Set new data to inputs and outputs
	 */
	public void set(SampleInputOutput sample){

    	//connect (copy) inputs (to the input edges)
    	String[] inputs = sample.getInput();
    	for(int i=0; i<inputs.length; i++){
    		double number = Double.parseDouble(inputs[i]);
    		inputNodes.get(i).setValue(number);
    	}

    	//connect (copy) outputs (to the output)
    	String[] outputs = sample.getOutput();
    	for(int i=0; i<outputs.length; i++){
    		double number = Double.parseDouble(outputs[i]);
    		outputNeurons.get(i).setDesiredValue(number);
    	}
	}	
	
	
	/**
	 * Get input neurons (actually not calculating nodes, we just reuse the same Neuron class)
	 */
	public ArrayList<Neuron> getInputNeurons(){
		return inputNodes;
	}
	
	
	/**
	 * Get output neurons (actually the last Layer)
	 */
	public ArrayList<Neuron> getOutputNeurons(){
		return outputNeurons;
	}
	

	/**
	 * For debugging purposes only.
	 * Recursive printing all the nodes (neurons) and edges starting from the inputs 
	 */
	public String toString(){
		StringBuffer sb = new StringBuffer();
		sb.append("Inputs: ");
    	Iterator<Neuron> it = inputNodes.iterator();
        while (it.hasNext()) {
        	Neuron neuron = it.next();
        	sb.append(neuron.getName());
        	if(it.hasNext())
        		sb.append(", ");        	
        }
		sb.append("\n");
		sb.append("Outputs: ");
    	it = outputNeurons.iterator();
        while (it.hasNext()) {
        	Neuron neuron = it.next();
        	sb.append(neuron.getName());
        	if(it.hasNext())
        		sb.append(", ");
        }
		sb.append("\n");
		return sb.toString();
	}

	
}
