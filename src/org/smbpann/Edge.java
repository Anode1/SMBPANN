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
 * Always points to some Node (Neurons), this is like pointer 
 */
public class Edge {

	private Neuron neuron; //corresponding to this Edge Neuron
	private String name; //for id purposes
	
	public double weight;
	public double weightDiff; //last weight difference
	public double dx; //error difference
	

	public Edge(Neuron neuron, String name){
		this.neuron=neuron;
		this.name=name;
		weight=Math.random(); //initialize by random (0..1) for now (we might want to have some flexibility here in future)
	}	

	
	/**
	 * Get corresponding Neuron 
	 */
	public Neuron getNeuron(){
		return neuron;
	}
	
	
	public String getName(){
		return name;
	}
	
	
	public void setName(String name){
		this.name=name;
	}
	
	/**
	 * For debugging purposes only
	 */
	public String toString(){
		return "name:"+name+", weight:"+weight+", dx:"+dx;
	}
	
}
