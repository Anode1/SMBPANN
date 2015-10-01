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
	
	private double step;
	public double currentError; //overall error
	public double goalError; //final error goal

	private ArrayList<Layer> layers; //main structure (list of Layers, each Layer is list of Neurons)
	
	
	public Network() throws Exception{
		step=hints.getAsDouble(Hints.STEP);
	}
	

	/**
	 * Create network using testingSet dimensions (used in lazy initialization)
	 */
	private void createNetwork(TestingSet testingSet) throws Exception{
		
		layers=new ArrayList<Layer>();
		
		//We build the network from the end (output) to the start (input): the number of outputs defines number of neurons
		//in the last (at least one) layer, as in perceptron. Depending on the number of hidden layers - we connect neurons
		//by edges until we connect the Input with all it's inputs to the 1st layer neurons

		Layer outputLayer=new Layer(); //at least one layer exists - even in the perceptron case
		layers.add(outputLayer);
		
		int nodeCounter=0; //we use numbers for naming neurons for id/debugging purposes for now
		
		//Connect output: we should have at least one output - to read the results from, so we always mapping. Number neurons=number of outputs 
		int sizeOfOutput = testingSet.getOutputSize();
		for(int i=0; i<sizeOfOutput; i++){
			Neuron neuron=new Neuron(Integer.toString(nodeCounter++)); 
			outputLayer.add(neuron);
		}
		
		//for now let's use previousLayer, later we'll have a dynamic variable policy here: how to choose Neurons to connect to
		Layer previousLayer=outputLayer;
		
		//decide how many hidden layers are necessary in the beginning. Use hints 
		
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
		
		
		
		//Finally connect all inputs with the last (previous) layer
		int sizeOfInput = testingSet.getInputSize();

		for(int i=0; i<sizeOfInput; i++){
	    	Iterator<Neuron> it = previousLayer.iterator();
	        while(it.hasNext()) {
	        	Neuron neuron = it.next();
	        	String name=neuron.getName()+"."+i; //name consists of name of Neuron, this Edge points to and number (for now)
	        	Edge edge=new Edge(neuron, name);
				neuron.addIncomingEdge(edge);
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
    	Iterator<InputOutput> it = testingSet.iterator();
        while (it.hasNext()) {
        	InputOutput test = it.next();
        	
        	//connect (copy) inputs (to the input edges)
        	Object[] inputs = test.getInput();

        	//connect (copy) outputs (to the output)
        	Object[] outputs = test.getOutput();
        	
        	
    		feedForward();
    		backPropagate();
        }
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
	
	
	/**
	 * For debugging purposes only.
	 * Recursive printing all the nodes (neurons) and edges starting from the inputs 
	 */
	public String toString(){
		StringBuffer sb = new StringBuffer();
		
		//print layers (with left edges for each Neuron, so we'll get all the network printed)
    	Iterator<Layer> layerIt = layers.iterator();
        while(layerIt.hasNext()) {
        	Layer layer = layerIt.next();
        	Iterator<Neuron> neuronIt = layer.iterator();
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
