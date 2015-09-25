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

public class Main{
	
	public static boolean trace, warning;

	/**
	 * non-GUI entry point (command-line)
	 */
    public static void main(String args[]) throws Exception{

		try{ 
			long t0=System.currentTimeMillis();

			new Config();
			
			/**
			 * If we don't accept input/output - it is online learning
			 */
			Network net=new Network();
			net.process();
			
			
			System.out.println("Processed in " + (System.currentTimeMillis()-t0) + " ms.");
		}
		catch(Throwable e){
			//log.error("", e);
			e.printStackTrace();
		}
 	}
    
}
