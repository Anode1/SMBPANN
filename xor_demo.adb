--  xor_demo.adb -- train a 2-4-1 network on XOR, the thesis's canonical
--  linearly-non-separable problem. The Ada counterpart of the C smbpann CLI's
--  built-in demo; same seed and hyper-parameters, so the two must agree.

with Ada.Text_IO;             use Ada.Text_IO;
with SMBPANN;                 use SMBPANN;
with SMBPANN.Rng;
with SMBPANN.Nets;            use SMBPANN.Nets;
with SMBPANN.Nets.Training;   use SMBPANN.Nets.Training;

procedure XOR_Demo is

   type Sample is record
      X : Real_Array (1 .. 2);
      D : Real_Array (1 .. 1);
   end record;

   Data : constant array (1 .. 4) of Sample :=
     (1 => (X => (0.0, 0.0), D => (1 => 0.0)),
      2 => (X => (0.0, 1.0), D => (1 => 1.0)),
      3 => (X => (1.0, 0.0), D => (1 => 1.0)),
      4 => (X => (1.0, 1.0), D => (1 => 0.0)));

   Epochs : constant := 20_000;

   Net : aliased Network := Create ((2, 4, 1));
   G   : SMBPANN.Rng.Generator;
begin
   SMBPANN.Rng.Seed (G, 1);
   Initialize (Net, G);

   Put_Line ("XOR  topology 2-4-1  weights" & Natural'Image (Num_Weights (Net))
             & "  rate 0.5  momentum 0.9  seed 1");

   declare
      T : Trainer := Create (Net'Access, 0.5, 0.9);
      E : Real;
   begin
      for Epoch in 1 .. Epochs loop
         E := 0.0;
         for S of Data loop
            E := E + Learn (T, S.X, S.D);
         end loop;
         if Epoch mod 2_000 = 1 then
            Put_Line ("  epoch" & Integer'Image (Epoch)
                      & "   sse" & Real'Image (E));
         end if;
      end loop;
   end;

   Put_Line ("learned:");
   for S of Data loop
      declare
         Y : constant Real_Array := Forward (Net, S.X);
      begin
         Put_Line ("  " & Real'Image (S.X (1)) & " XOR" & Real'Image (S.X (2))
                   & "  ->" & Real'Image (Y (1))
                   & "   (target" & Real'Image (S.D (1)) & ")");
      end;
   end loop;
end XOR_Demo;
