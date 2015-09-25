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

public class Neuron {

	private double value;
	private double bias;
	private ArrayList<Edge> edges=new ArrayList<Edge>();
	
	public Neuron(){
		
	}
	
	
	public void addEdge(Edge edge){
		edges.add(edge);
	}
	
	
	public double getOutput(){
		return value;
	}
	
	
	/**
	 * For debugging purposes only
	 */
	public String toString(){
		StringBuffer sb=new StringBuffer();
    	Iterator<Edge> it = edges.iterator();
        while (it.hasNext()) {
        	Edge edge = it.next();
        	sb.append(edge);
        	if(it.hasNext())sb.append(",");
        }
        sb.append("\n");
        sb.append("output="+Double.toString(value));
		return sb.toString();
	}	
	
}
