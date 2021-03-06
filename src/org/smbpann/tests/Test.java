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
package org.smbpann.tests;

import java.util.ArrayList;

import org.smbpann.*;

import junit.framework.TestCase;

/**
 * This is a sandbox. Move working test into Tests or another TestSuite after 
 * completion of the work unit 
 */
public class Test extends TestCase{

    
    public void setUp() throws Exception{
    	new Config();
    	
    	//Main.trace=true;
    }
    
    

    
    
    public void testBooleanOR() throws Exception{
   	
    	Network net=new Network();
    	net.getHints().put(Hints.NUMBER_OF_HIDDEN_LAYERS, 0);
    	
    	TestingSet testingSet=new TestingSet();
    	testingSet.add(new SampleInputOutput(new String[]{"0", "0"}, new String[] {"0"}));
    	testingSet.add(new SampleInputOutput(new String[]{"1", "0"}, new String[] {"0"}));
    	testingSet.add(new SampleInputOutput(new String[]{"0", "1"}, new String[] {"0"}));
    	testingSet.add(new SampleInputOutput(new String[]{"1", "1"}, new String[] {"1"}));
    	
		net.teach(testingSet); //one pass
		
    	System.out.println(net);
		
    }
  
    
    /**
     * Percepton should not work
     */
    public void testBooleanXOR() throws Exception{

    	Network net=new Network();
    	
    	TestingSet testingSet=new TestingSet();
    	testingSet.add(new SampleInputOutput(new String[]{"0", "0"}, new String[] {"0"}));
    	testingSet.add(new SampleInputOutput(new String[]{"1", "0"}, new String[] {"1"}));
    	testingSet.add(new SampleInputOutput(new String[]{"0", "1"}, new String[] {"1"}));
    	testingSet.add(new SampleInputOutput(new String[]{"1", "1"}, new String[] {"0"}));
    	
		net.teach(testingSet);
		
    }
 

}
