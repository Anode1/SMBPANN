#!/bin/sh

CP=.
CP=${CP}:./classes
CP=${CP}:./smbpann.jar

# convert the unix path to windows
if [ "$OSTYPE" = "cygwin32" ] || [ "$OSTYPE" = "cygwin" ] || [ "$OS" = "Windows_NT" ] ; then
  CP=`cygpath --path --windows "${CP}"`
fi

#Set JVM parameters here - such as -Xmx1024 etc
#memory is taken by centimorgans map (other operations are done on streams and state machine)
java -Xms128m -Xmx512m -classpath "${CP}:${CLASSPATH}" org.smbpann.Main "$@"

