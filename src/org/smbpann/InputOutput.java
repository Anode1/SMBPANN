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
 * Placeholder for one epoch training sample: input + required output (it may be as complex as image, 
 * or a multi-dimensional array in future)
 */
public class InputOutput {
	
	private Object[] inputs; //for now it is an array - will be changed later. This class will hide the implementation
	private Object[] desiredOutput; //for now it is an array - will be changed later. This class will hide the implementation

	
	public InputOutput(Object[] inputs, Object[] desiredOutput){
		this.inputs=inputs;
		this.desiredOutput=desiredOutput;
	}
	
	
	public Object[] getInput(){
		return inputs;
	}
	
	
	public Object[] getOutput(){
		return desiredOutput;
	}
	
	
	public int getInputSize(){
		return inputs.length;
	}
	
	
	public int getOutputSize(){
		return desiredOutput.length;
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
		sb.append(" ==> ");
		for(int i=0; i<desiredOutput.length; i++){
			sb.append(desiredOutput[i]);
			if(i<desiredOutput.length-1)
				sb.append(",");
		}	
		sb.append("\n");
		return sb.toString();
	}
}
