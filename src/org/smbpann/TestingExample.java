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

public class TestingExample {

	private Input input;
	private Output output;
	
	
	public TestingExample(Input input, Output output){
		this.input=input;
		this.output=output;
	}
	
	/**
	 * Convenience constructor for simple string inputs used in simple tests
	 */
	public TestingExample(String[] inputs, String[] outputs){
		this.input=new Input(inputs);
		this.output=new Output(outputs);
	}
	
	
	public long getInputSize(){
		return input.getSize();
	}
	
	
	public long getOutputSize(){
		return output.getSize();
	}
	
	
	/**
	 * For debugging purposes only
	 */
	public String toString(){
		StringBuffer sb = new StringBuffer("testing example: ");
		sb.append(input);
		sb.append(" ==> ");
		sb.append(output);
		sb.append("\n");
		return sb.toString();
	}
}
