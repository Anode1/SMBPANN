/*
 	Copyright (C) 2009 Vasili Gavrilov

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
import java.util.Properties;

/**
 * Parser for arguments passed through command line implementing POSIX arguments
 */
public class ArgsParser{
	
    /**
     * Convenience utility function - for tests through API (to reuse the same
     * arguments as used in command-line version). This method is to be used
     * through API, and it is only simulating of POSIX argument, so throw 
     * exception if something wrong (it is not user input - as in Main!)
     */
	public static void setParameters(String[] args, Properties argsAsProps) throws Exception{
		if(new ArgsParser().setArgs(args, Parameters.getInstance()) != 0){
			throw new Exception("Something wrong with parameters"); 
		}
	}
	
	
	int setArgs(String[] args, Properties argsAsProps){

		/*  //if we require to pass something:
	    if(args==null || args.length==0){
	      printUsage();
	      return -1;
	    }
		 */

		for(int i = 0; i < args.length; i++){

			String arg = args[i];
			if(arg==null)continue;

			arg=arg.trim();

			//System.out.println("Processing of argument:"+arg);

			if(arg.startsWith("-")){//treat as an option

				if(arg.startsWith("-D")){ //if we'll want it in java way (no, we want POSIX!!!)
					if(treatAsAProperty(arg, i, args, argsAsProps, Config.isWindows)==-1){
						return -1;
					}
				}
				else if (arg.equals("-h") || arg.equals("--help")){
					printUsage();
					return -1;
				}
				else if (arg.equals("-t") || arg.equals("--trace")){
					//Logger rootLogger = logger.getRootLogger();
					//rootLogger.setLevel(Level.TRACE);
					Main.trace=true;
				}				
				else if (arg.equals("-i") || arg.equals("--input")){
					i++;
					if(i>=args.length){
						System.out.println("Option "+args[i-1]+" requires argument");
						System.out.println();
						return -1;
					}
					ArrayList listOfFilesCollected = (ArrayList)argsAsProps.get(Constants.INPUT_FILE_KEY);
					if(listOfFilesCollected==null){
						listOfFilesCollected = new ArrayList();
						argsAsProps.put(Constants.INPUT_FILE_KEY, listOfFilesCollected);
					}
					listOfFilesCollected.add(args[i]);
				}
				else if (arg.equals("-o") || arg.equals("--output")){
					i++;
					if(i>=args.length){
						System.out.println("Option "+args[i-1]+" requires argument");
						System.out.println();
						return -1;
					}					
					argsAsProps.put(Constants.OUTPUT_FILE_KEY, args[i]);
				}
				else if (arg.equals("-d") || arg.equals("--dir")){
					i++;
					if(i>=args.length){
						System.out.println("Option "+args[i-1]+" requires argument");
						System.out.println();
						return -1;
					}					
					argsAsProps.put(Constants.OUTPUT_DIR_KEY, args[i]);
				}				
				else if (arg.equals("--threshold")){
					i++;
					if(i>=args.length){
						System.out.println("Option "+args[i-1]+" requires argument");
						System.out.println();
						return -1;
					}					
					argsAsProps.put(Constants.THRESHOLD_KEY, args[i]);
				}
				else if (arg.equals("--someflag")){
					//argsAsProps.put(Constants.SHOW_SIMILARITY, "true");
				}
				else if (arg.equals("-W")){
					Main.warning=false;
				}
				else{
					System.out.println("Argument: "+arg+" not supported");
					System.out.println();
					return -1;
				}
			}
			else{ //single argument for now
				//argsAsProps.put(Constants.INPUT_FILE_KEY, arg); 
				
				ArrayList listOfFilesCollected = (ArrayList)argsAsProps.get(Constants.INPUT_FILE_KEY);
				if(listOfFilesCollected==null){
					listOfFilesCollected = new ArrayList();
					argsAsProps.put(Constants.INPUT_FILE_KEY, listOfFilesCollected);
				}
				listOfFilesCollected.add(args[i]);				
				
			}
		}//for args
		
		//
		//Validation of contradictory things:
		//

		return 0;
	}//setArgs

	
	/**
	 * Try to parse an argument as an overriden property.
	 * This method is used by setArgs only and the last one is not used
	 */
	private int treatAsAProperty(String arg, int i, String[] args, 
			Properties argsAsProps, boolean dos){

		//System.out.println("treating arg:"+arg);

		if(dos){
			if(i+1>=args.length){
				System.out.println("property value cannot be null");
				printPropsUsage();
				return -1;
			}

			String key=arg.substring(2).trim();

			if("".equals(key)){
				System.out.println("property key cannot be null");
				printPropsUsage();
				return -1;
			}

			String value=args[i+1];
			if(value==null){
				System.out.println("property value cannot be null");
				printPropsUsage();
				return -1;
			}

			value=value.trim();
			argsAsProps.put(key,value);
			i++; //in dos '=' considered to be a delimiter
		}
		else{
			String keyAndValue=arg.substring(2).trim();

			if("".equals(keyAndValue)){
				//System.out.println("Error 1");
				printPropsUsage();
				return -1;
			}

			int indexOfEqualSign=keyAndValue.indexOf("=");
			if(indexOfEqualSign==-1){
				//System.out.println("Error 2");
				printPropsUsage();
				return -1;
			}
			String key=keyAndValue.substring(0,indexOfEqualSign).trim();
			String value=keyAndValue.substring(indexOfEqualSign+1).trim();
			argsAsProps.put(key,value);
			//System.out.println("Property added:"+key+"="+value);
		}
		return 0;
	}


	/**
	 * Print usage for properties. Not used now - when getOpt() is used
	 */
	private void printPropsUsage(){

		System.out.println("Usage for properties: -Dproperty=value");
	}  

	
	/**
	 * Print command-line usage help screen to stdout
	 */
	private void printUsage(){
		  
	   String progName="smbpann"; 
	   
	   System.out.println("");
	   System.out.println(" " + progName+" - Self-modifying Neural Network "+Constants.releaseVersionString);
	   System.out.println(" Copyright (C) 2015");
	   System.out.println(" This program comes with ABSOLUTELY NO WARRANTY;");
	   System.out.println(" This is free software, and you are welcome to redistribute it");
	   System.out.println(" under certain conditions; see <http://www.gnu.org/licenses/>.");
	   
	   System.out.println();
	   System.out.println("Usage: ");

	   System.out.println("     "+progName+" --input 1.txt");
	   System.out.println("                     Process one file (see below - which processor is invoked");
	   
	   System.out.println("     "+progName+" -i 1.txt");
	   System.out.println("                     The same as previous but shorter");

	   System.out.println("     "+progName+" 1.txt");
	   System.out.println("                     The shortest way to pass a file");
	   
	   System.out.println("     "+progName+" \"Some Directory\"");
	   System.out.println("                     All files in passed directory will be processed");

	   System.out.println("     "+progName+" -i /home/vasya/file1.txt -i /home/pasha/file2.txt -i All");
	   System.out.println("                     Passing list of files (mixing with directories allowed)");   
	   
	   System.out.println("");
	   System.out.println("Options (FIXME!):");
	   System.out.println("");
	   System.out.println(" -h, --help         This help screen");
	   System.out.println(" -t, --trace        Pass output file path");
	   System.out.println(" -i, --input        Pass file (absolute or relative path)");
	   System.out.println("     (files can also be passed as arguments without keys)");
	   System.out.println(" -o, --output       Pass output file path");
	   System.out.println(" -u  --hir          HIR Calculator mode");
	   System.out.println("      Options working with HIR Calculator only:");
	   System.out.println("         -cM        Centimorgans threshold");
	   System.out.println("         -m         Minimal number of snips threshold");
	   System.out.println("         --cg       Show also similarity of 2 genomes, using 1-0.5-metrics");
	   //System.out.println("         --cMf      File path to centimorgans mapping");	   
	   System.out.println(" --hapmap           Use HapMap file(s) as input(s)");
	   System.out.println(" -z,     --roh      Show regions and percentage of homozygousity");
	   System.out.println("      Options working with homozygousity calculator:");	   
	   System.out.println("         -m         Minimal number of snips threshold");
	   System.out.println("         -nc        Treat no-calls are heterozygous");
	   System.out.println(" --cg               Show similarity between 2 genomes, using 1-0.5-metrics");
	   System.out.println(" --cg2              Show similarity between 2 genomes, using 1-0.75-metrics");
	   //System.out.println(" --cg3              Show similarity between 2 genomes, using 1, 0.75 x cM metrics");
	   System.out.println(" -M  --merge        Merge 2 genotypes into one file");
	   System.out.println(" -W                 Suppress warnings");
	   System.out.println(" -p                 ped2raw transformation ");
	   System.out.println("      Options working with ped2raw only:");
	   System.out.println("         -d provide directory for storing files");
	   System.out.println("                      (requires --snips in form: \"chr snip 0 distance\")");
	   System.out.println(" -b                 Transforming Behar series-matrix files into PED (not completed)");
	   System.out.println(" --phase            Phase two FTDNA files");
	   System.out.println("");
	}


	
	public static void main(String[] args){

		try{

			new Config(); //configure from system.properties and log4j properties  

			if(new ArgsParser().setArgs(args, Parameters.getInstance()) != 0){
				return; //either we have an error in arguments or help asked to be printed
			}

			Parameters.getInstance().list(System.out);
		} 
		catch(Throwable e){
			e.printStackTrace();
		}
	}  

}

