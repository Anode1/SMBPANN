package org.smbpann;

import java.util.Properties;

public class Hints extends Properties{
	
	public static final String NUMBER_OF_HIDDEN_LAYERS 	= "number_of_hidden_layers";
	public static final String STEP 					= "step";
	
	
	public String getString(String key){

		String value=getProperty(key);
		return (value!=null?value.trim():null);
	}
	
	
	public int getAsInteger(String key){

		String i = Parameters.getString(key);

		try{
			return Integer.parseInt(i);
		}catch(NumberFormatException e){
			System.err.println("Malformed integer in properties for key="+key +":"+i);
		}
		return 0;
	}
	
	
	public double getAsDouble(String key){

		String i = Parameters.getString(key);

		try{
			return Double.valueOf(i).doubleValue();
		}catch(NumberFormatException e){
			System.err.println("Malformed double in properties for key="+key +":"+i);
		}
		return 0.0;
	}
	

	public boolean getAsBoolean(String key){

		String s = Parameters.getString(key);
		return str2Bool(s);
	}

	
	public boolean str2Bool(String s){

		if(s==null)return false;

		if(s.equals("true"))return true; //getString() is already trimmed
		if(s.equals("false"))return false;
		s=s.toLowerCase();
		if(s.equals("true"))return true;

		return false;
	}
	
}
