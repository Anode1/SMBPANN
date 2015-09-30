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
 * Testing/Training set of individual examples (a vector)
 */
public class TestingSet extends ArrayList<InputOutput>{

	/**
	 * Returns Input size of the first element (all examples have the same size) 
	 */
	public int getInputSize() throws Exception{
		if(this.size()==0) throw new Exception("There are no testing examples in the testing set");
		InputOutput firstTest=this.get(0);
		return firstTest.getInputSize();
	}

	
	/**
	 * Returns Output size of the first element (all examples have the same size) 
	 */
	public int getOutputSize() throws Exception{
		if(this.size()==0) throw new Exception("There are no testing examples in the testing set");
		InputOutput firstTest=this.get(0);
		return firstTest.getOutputSize();
	}
	
	
	/**
	 * For debugging purposes only
	 */
	public String toString(){
		StringBuffer sb = new StringBuffer("testing set:\n");
    	Iterator<InputOutput> it = this.iterator();
        while (it.hasNext()) {
        	InputOutput test = it.next();
        	sb.append(test);
        }
		return sb.toString();
	}
}
