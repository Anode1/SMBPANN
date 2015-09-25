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

public class Input {
	
	private Object[] inputs;
	
	public Input(Object[] inputs){
		this.inputs=inputs;
	}
	
	
	public long getSize(){
		return inputs.length;
	}
	
	
	/**
	 * For debugging purposes only
	 */
	public String toString(){
		StringBuffer sb = new StringBuffer();
		for(int i=0; i<inputs.length; i++){
			sb.append(inputs[i]);
			if(i<inputs.length-1)
				sb.append(",");
		}
		return sb.toString();
	}
}
