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

	public String name; //name of the network for identification (when running multiple)
	private Hints hints=new Hints(); //way to pass heuristics to the network - when it is necessary (the proper function will get it when it needs, depending on the particular function)
	
	public static double learningRate;
	public static double momentum;
	public static double weigthDecay; 
	public static double goalError; //final error goal

	public double currentError; //overall error
	
	private ArrayList<Layer> layers; //main structure (list of Layers, each Layer is list of Neurons; input is not a layer)
	private InputOutputTerminals inputOutputTerminals; //input/output terminals, for easy and fast data setting
	
	
	public Network() throws Exception{
		learningRate=Parameters.getAsDouble("learning_rate");
		momentum=Parameters.getAsDouble("momentum");
		weigthDecay=Parameters.getAsDouble("weight_decay");	
		goalError=Parameters.getAsDouble("error");
	}
	

	/**
	 * Create network using testingSet dimensions (used in lazy initialization)
	 */
	private void createNetwork(TestingSet testingSet) throws Exception{
		
		//create processing layers (which consist of hidden layers and at least one output layer) 
		layers=new ArrayList<Layer>();
		
		inputOutputTerminals = new InputOutputTerminals(testingSet);
		
		//we create Layers from Output to Input backwards
		
		Layer outputLayer=new Layer();
		outputLayer.setNeurons(inputOutputTerminals.getOutputNeurons()); //set reference to the same list of Neurons
		layers.add(outputLayer);
		
		int nodeCounter=0; //we use numbers for naming neurons for id/debugging purposes for now

		Layer previousLayer=outputLayer; //set output as previousLayer, later we'll have a dynamic variable policy here: how to choose Neurons to connect to
		
		//decide how many hidden layers are necessary in the beginning. Use hints (?) 
		
		//TODO: implement hidden layers and interconnection here. In future we might experiment with trimming edges online
		// . . .
		/*
		int numberOfHiddenLayers=3; //FIXME! And no layers in general case
		for(int i=0; i<numberOfHiddenLayers; i++){
		
			Layer hiddenLayer=new Layer();
			layers.add(0, hiddenLayer); //push to the front - to have the list order from left to right
			
			//how many nodes in each layer (everything here will be rewritten: to have it dynamic: we just show the idea)
			int numberOfNodesPerLayer=10;
			for(int j=0; j<numberOfNodesPerLayer; j++){
				Neuron neuron=new Neuron(Integer.toString(nodeCounter++));
				hiddenLayer.add(neuron);
				//connect newly created Neuron with each neuron of the previous layer (for now, in future we might want to create Edges/connectino separately)
				 
			}
			
		}
		*/
		
		//Finally connect all neurons of just created (previous) layer to all the Inputs
		
    	Iterator<Neuron> it = previousLayer.getNeurons().iterator();
        while (it.hasNext()) {
        	Neuron neuron = it.next();
        	Iterator<Neuron> inputIt = inputOutputTerminals.getInputNeurons().iterator();
        	while (inputIt.hasNext()) {
        		Neuron inputNeuron = inputIt.next();
        		Edge edge=new Edge(inputNeuron, neuron); //this connects itself with both passed Neurons
        	}
        }

	
	}
	
	
	//Dynamic procedures for insertion/deleting neurons online (removes from Layers etc)
	
	public void insertEdge(Edge edge, Neuron fromNode, Neuron toNode){
		
	}
	
	
	public void removeEdge(Edge edge){
		
	}
	
	
	public void insertNeuron(Neuron fromNode, Layer layer){
		
	}
	
	
	public void removeNeuron(Edge edge){
		
	}

	
	
	/**
	 * One iteration: feed-forward and back propagation 
	 */
	public void teach(TestingSet testingSet) throws Exception{

		if(Main.trace) System.out.println(testingSet);
		
		//lazy initialization of the Network using the testing set dimensions
		if(layers==null){
			createNetwork(testingSet);
		}
		
		//iterating through the testing set:
    	Iterator<SampleInputOutput> it = testingSet.iterator();
        while (it.hasNext()) {
        	SampleInputOutput sample = it.next();
        	inputOutputTerminals.set(sample);        	
        	
    		feedForward();
    		backPropagate();
        }
	}
	
	
	public void feedForward() throws Exception{
    	Iterator<Layer> layerIt = layers.iterator();
        while(layerIt.hasNext()) {
        	Layer layer = layerIt.next();
        	Iterator<Neuron> neuronIt = layer.getNeurons().iterator();
        	while(neuronIt.hasNext()){
        		Neuron neuron=neuronIt.next();
        		neuron.feedForward();
        	}
        }
	}
	
	
	public void backPropagate() throws Exception{
		int lastLayerIndex = layers.size()-1;
		for(int i=lastLayerIndex; i>=0; i--){ //from right (last year) to left (to the input)
			boolean isLastLayer=(i==lastLayerIndex);
        	Layer layer=layers.get(i);
        	Iterator<Neuron> neuronIt = layer.getNeurons().iterator();
        	while(neuronIt.hasNext()){
        		Neuron neuron=neuronIt.next();
        		neuron.backPropagate(isLastLayer);
        	}
        }
	}
	
	
	public double getCurrentError(){
		return currentError;
	}
	
	
	public Hints getHints(){
		return hints;
	}
	
	
	/**
	 * For debugging purposes only.
	 * Recursive printing all the nodes (neurons) and edges starting from the inputs 
	 */
	public String toString(){
		StringBuffer sb = new StringBuffer();
		
		Iterator<Neuron> it = inputOutputTerminals.getInputNeurons().iterator();
		while(it.hasNext()) {
			Neuron neuron=it.next();
			neuron.getOutputValue();
		}
		
		//print layers (with left edges for each Neuron, so we'll get all the network printed)
    	Iterator<Layer> layerIt = layers.iterator();
        while(layerIt.hasNext()) {
        	Layer layer = layerIt.next();
        	Iterator<Neuron> neuronIt = layer.getNeurons().iterator();
        	while(neuronIt.hasNext()){
        		Neuron neuron=neuronIt.next();
        		ArrayList<Edge> incomingEdges = neuron.getIncomingEdges();
        		Iterator<Edge> edgesIt = incomingEdges.iterator();
        		while(edgesIt.hasNext()){
        			Edge edge=edgesIt.next();
        			sb.append(edge); sb.append("\n"); //print edges first
        		}
        		sb.append(neuron); sb.append("\n"); //print neuron after edges
        	}
        }
		
		sb.append("\n");
		return sb.toString();
	}

}
