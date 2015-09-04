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

import junit.framework.TestCase;


/**
 * List of all unit tests. Please - run this list after any change and fix
 * (failure in this list means that something is broken by the change!)
 * 
 * TODO: write more regression tests! For all the processes and all different 
 * cases !
 */
public class Tests extends TestCase{

    public Tests(String name){
        super(name);
    }

    public void setUp(){

    }
    
    
 /*
    
    public void testRAW2PED() throws Exception{
    	
    	File output = new File(Constants.DEFAULT_OUTPUT_FILE);
    	output.delete();
    	
		ArrayList filesList = new ArrayList();
		filesList.add("data/1.txt");
		Parameters.getInstance().put(Constants.INPUT_FILE_KEY, filesList);    	
    	
    	MainProcessor processor = new MainProcessor();
		assertFalse(processor.init().thereAreErrors());
		processor.process();
		//System.out.println(TestUtils.file2String(Constants.DEFAULT_OUTPUT_FILE).trim());
		assertTrue("1.txt 1 1 1 1 1 G G  0 0  A A  0 0  0 0  0 0  0 0  0 0  0 0  T A  0 0  0 0  0 0  0 0  0 0".equals(TestUtils.file2String(Constants.DEFAULT_OUTPUT_FILE).trim()));
	
		assertTrue(output.delete());
		
		filesList = new ArrayList();
		filesList.add("data/2.txt");
		Parameters.getInstance().put(Constants.INPUT_FILE_KEY, filesList);   		
		processor.process();
		assertTrue("2.txt 1 1 1 1 1 0 0  A G  T T  0 0  0 0  0 0  0 0  0 0  0 0  0 0  0 0  0 0  0 0  0 0  0 0".equals(TestUtils.file2String(Constants.DEFAULT_OUTPUT_FILE).trim()));		
		
		assertTrue(output.delete());
		
		filesList = new ArrayList();
		filesList.add("data/3.txt");
		Parameters.getInstance().put(Constants.INPUT_FILE_KEY, filesList);   		
		processor.process();
		assertTrue("3.txt 1 1 1 1 1 G A  0 0  0 0  0 0  0 0  0 0  0 0  0 0  0 0  0 0  0 0  0 0  0 0  0 0  0 0".equals(TestUtils.file2String(Constants.DEFAULT_OUTPUT_FILE).trim()));		
		
		assertTrue(output.delete());
		
		filesList = new ArrayList();
		filesList.add("data/4.txt");
		Parameters.getInstance().put(Constants.INPUT_FILE_KEY, filesList);   		
		processor.process();
		assertTrue("4.txt 1 1 1 1 1 0 0  A G  0 0  0 0  0 0  0 0  0 0  0 0  0 0  0 0  0 0  0 0  0 0  0 0  0 0".equals(TestUtils.file2String(Constants.DEFAULT_OUTPUT_FILE).trim()));		
				
		assertTrue(output.delete());
		
		filesList = new ArrayList();
		filesList.add("data/5.txt");
		Parameters.getInstance().put(Constants.INPUT_FILE_KEY, filesList);   		
		processor.process();
		assertTrue("5.txt 1 1 1 1 1 0 0  0 0  T T  0 0  0 0  0 0  0 0  0 0  0 0  0 0  0 0  0 0  0 0  0 0  0 0".equals(TestUtils.file2String(Constants.DEFAULT_OUTPUT_FILE).trim()));		
				
		output.delete();
		processor.finish();
    }
    
    	
     public void testSome() throws Exception{
    	
    	//multiple chromosomes:
    	TestUtils.string2File(TEMPFILE1,
    			"i1000001 1 1 GG" + Constants.NL +
    			"i1000003 2 3 AA");
    	TestUtils.string2File(TEMPFILE2,
    			"i1000001 1 1 GG" + Constants.NL +
    			"i1000002 2 2 AG");
    	Main.main(new String[]{"-M", TEMPFILE1, TEMPFILE2, "-o", RESULTFILE, "-W"});
    	//System.out.println(TestUtils.file2String(RESULTFILE));
    	assertTrue(
    		TestUtils.file2String(RESULTFILE).equals(
   				"i1000001	1	1	GG" + Constants.NL +
    			"i1000002	2	2	AG" + Constants.NL +
    			"i1000003	2	3	AA" + Constants.NL    			
    		)
    	);
    	
    }  
*/
}