@echo off
REM if no release jar - just look into classes (easier during development)
set CP=.
set CP=%CP%;./classes
set CP=%CP%;./smbpann.jar

REM If jre provided (included) - just drop the whole "jre" into the project directory, it will not hurt
set PATH=%PATH%;.\jre\bin\

java -Xms128m -Xmx512m -classpath "%CP%;%CLASSPATH%" org.smbpann.Main %1 %2 %3 %4 %5 %6 %7 %8 %9

