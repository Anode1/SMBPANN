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
 * Grouping of Neurons where those in the same layers can be calculated in-parallel at the same time and independently (not connected to each other)  
 */
public class Layer extends ArrayList<Neuron>{

	
	/**
	 * For debugging purposes only
	 */
	public String toString(){
		StringBuffer sb=new StringBuffer();
    	Iterator<Neuron> it = this.iterator();
        while (it.hasNext()) {
        	Neuron neuron = it.next();
        	sb.append(neuron); sb.append("\n");
        }
		return sb.toString();
	}	
}
