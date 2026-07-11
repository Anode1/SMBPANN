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

import java.util.Properties;

public class Hints extends Properties{
	
	public static final String NUMBER_OF_HIDDEN_LAYERS 	= "number_of_hidden_layers";
	
	
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
