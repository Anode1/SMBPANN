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

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.HashMap;


public class FileUtil {
	
	
	protected static void processDir(String filesDir) throws Exception{

		BufferedWriter writer = null;
		try{
			writer=new BufferedWriter(new FileWriter(Config.getOutputFile(), true));			
			
			File mainDir = new File(filesDir);
		    int entries = mainDir.list().length;
	
		    String[] level1files = mainDir.list();
		    
   
			//if(Main.progressListener!=null)Main.progressListener.start(entries); //show progress
			
		    for(int i = 0; i < entries; i++){
		    	String filePath=filesDir + File.separator + level1files[i];
		        File aFile = new File(filePath);
		        
		        if(aFile.isDirectory()){
		        	continue; //skip directories
		        }
	        	else if(aFile.isFile()){
	        		processFile(filePath, writer);
	        		//if(Main.progressListener!=null)Main.progressListener.updateStatus();
	        	}
		    }
		    
		    //if(Main.progressListener!=null)Main.progressListener.finish(); //hide progress bar
		    
		}
		finally{
			if(writer!=null)try{writer.close();}catch(Exception e){}
		}
	}
	
	
	public static void processFile(String filePath, BufferedWriter writer) throws Exception{
		
		BufferedReader reader=null;
		try{
			File file = new File(filePath);
			if(!file.exists() || !file.canRead()){
				throw new Exception("Cannot read file");
			}

			reader = new BufferedReader(new FileReader(file));

			/*
			while((record=reader.getNextRecord())!=null){
				
				lineCounter++;
				
				if(!record.chromosome.equals(prevChromo)){
					if(prevChromo!=null){
						onEndChromo(prevChromo);
					}
					onStartChromo(lineCounter, record, prevChromo, record.chromosome);
				}
										
				checkBase(lineCounter, record, prevChromo);
			
				prevChromo=record.chromosome;				
			}
			
			 */
		}
		finally{
			if(reader!=null)try{reader.close();}catch(Exception e){}
		}
	}
	
	
    /**
     * Finds the first line starting with string, returning the whole line 
     */
    public static String extractLineStartWith(File file, String startsWith) throws Exception{
		BufferedReader reader = null;
		try{
			reader = new BufferedReader(new FileReader(file));
			String line;
			while((line = reader.readLine())!=null){
				if(line.startsWith(startsWith)){
					return line;
				}
			}
			return null;
		}
		finally{
			if(reader!=null)try{reader.close();}catch(Exception e){}
		}
    }
    

    /**
     * Finds the lines starting with string, returning the Hash 
     */
    public static HashMap extractLines(File file, String[] keys) throws Exception{
    	HashMap results= new HashMap();
		BufferedReader reader = null;
		try{
			reader = new BufferedReader(new FileReader(file));
			String line;
			while((line = reader.readLine())!=null){
				
				for(int i=0; i<keys.length; i++){
				
					if(line.startsWith(keys[i])){
						results.put(keys[i], line);
					}
				}
			}
			return results;
		}
		finally{
			if(reader!=null)try{reader.close();}catch(Exception e){}
		}
    }
    
    
}