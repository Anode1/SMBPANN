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
 * Creates more than one Network 
 */
public class Processor {
	
	private boolean terminating;
	

	public void process(Network net) throws Exception{
		try{
			while(!terminating){
				net.fire();
				
				//terminate if threshold reached
				if(net.currentError < net.goalError){
					break;
				}
			}
		}
		finally{
			if(Main.trace) System.out.println("stopped");
		}
	}

	
	public void stop(){
		terminating=true;
	}	
	
}
