--  smbpann-data.ads -- plain-text datasets and a train/test split.
--
--  A Dataset is loaded once from a whitespace-separated text file: each
--  non-blank, non-'#' line holds N_Input input values followed by N_Output
--  target values. Values live in two flat arrays (all inputs, all targets),
--  allocated once and freed automatically (a controlled type). A Split is a
--  shuffled permutation of the sample indices partitioned into a training and a
--  test part; it holds only the ordering, never a copy of the data.

with Ada.Finalization;
with SMBPANN.Rng;

package SMBPANN.Data is

   Data_Error : exception;

   type Dataset is tagged limited private;

   --  Load PATH. Raises Data_Error on an unreadable file or a line whose field
   --  count is not N_Input + N_Output.
   function Load (Path : String; N_Input, N_Output : Positive) return Dataset;

   function Samples  (D : Dataset) return Natural;
   function N_Input  (D : Dataset) return Positive;
   function N_Output (D : Dataset) return Positive;

   --  Input / target row of sample I (1 .. Samples).
   function Input  (D : Dataset; I : Positive) return Real_Array
     with Pre => I <= Samples (D);
   function Target (D : Dataset; I : Positive) return Real_Array
     with Pre => I <= Samples (D);

   type Split is tagged limited private;

   --  Shuffle the sample indices with G and take the first Fraction as training.
   function Make_Split
     (D : Dataset'Class; Fraction : Real; G : in out SMBPANN.Rng.Generator)
      return Split
     with Pre => Fraction >= 0.0 and then Fraction <= 1.0;

   function Train_Count (S : Split) return Natural;
   function Test_Count  (S : Split) return Natural;

   --  The sample index at position P within the training (resp. test) part.
   function Train_Index (S : Split; P : Positive) return Positive
     with Pre => P <= Train_Count (S);
   function Test_Index  (S : Split; P : Positive) return Positive
     with Pre => P <= Test_Count (S);

private

   type Real_Array_Access is access Real_Array;

   type Dataset is new Ada.Finalization.Limited_Controlled with record
      NI, NO : Positive := 1;
      N      : Natural  := 0;
      X      : Real_Array_Access;   --  N * NI, row-major
      Y      : Real_Array_Access;   --  N * NO, row-major
   end record;

   overriding procedure Finalize (D : in out Dataset);

   type Index_Array is array (Positive range <>) of Positive;
   type Index_Access is access Index_Array;

   type Split is new Ada.Finalization.Limited_Controlled with record
      N       : Natural := 0;
      N_Train : Natural := 0;
      Order   : Index_Access;       --  a permutation of 1 .. N
   end record;

   overriding procedure Finalize (S : in out Split);

end SMBPANN.Data;
