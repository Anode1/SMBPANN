--  smbpann-data.adb -- dataset loading and the train/test split. See the spec.

with Ada.Text_IO;             use Ada.Text_IO;
with Ada.Text_IO.Unbounded_IO;
with Ada.Strings.Unbounded;   use Ada.Strings.Unbounded;
with Ada.Unchecked_Deallocation;
with Interfaces;              use type Interfaces.Unsigned_32;

package body SMBPANN.Data is

   procedure Free is
     new Ada.Unchecked_Deallocation (Real_Array, Real_Array_Access);
   procedure Free is
     new Ada.Unchecked_Deallocation (Index_Array, Index_Access);

   --  A line carries data unless it is blank or its first non-blank is '#'.
   function Is_Data (Line : String) return Boolean is
      I : Natural := Line'First;
   begin
      while I <= Line'Last
        and then (Line (I) = ' ' or else Line (I) = ASCII.HT)
      loop
         I := I + 1;
      end loop;
      return I <= Line'Last and then Line (I) /= '#';
   end Is_Data;

   --  Fill Vals with the whitespace-separated numbers on Line; return how many
   --  there were (which the caller checks against N_Input + N_Output).
   procedure Parse (Line : String; Vals : out Real_Array; Count : out Natural) is
      I : Natural := Line'First;
      N : Natural := 0;
   begin
      loop
         while I <= Line'Last
           and then (Line (I) = ' ' or else Line (I) = ASCII.HT)
         loop
            I := I + 1;
         end loop;
         exit when I > Line'Last;
         declare
            Start : constant Natural := I;
         begin
            while I <= Line'Last
              and then Line (I) /= ' ' and then Line (I) /= ASCII.HT
            loop
               I := I + 1;
            end loop;
            N := N + 1;
            if N <= Vals'Length then
               declare
                  --  Ada's 'Value wants a real literal ("0.0"), so append ".0"
                  --  to a plain integer token ("0") to accept integer-valued data.
                  Tok  : constant String := Line (Start .. I - 1);
                  Real_Lit : Boolean := False;
               begin
                  for C of Tok loop
                     if C = '.' or else C = 'e' or else C = 'E' then
                        Real_Lit := True;
                     end if;
                  end loop;
                  Vals (Vals'First + N - 1) :=
                    Real'Value (if Real_Lit then Tok else Tok & ".0");
               end;
            end if;
         end;
      end loop;
      Count := N;
   exception
      when others =>
         raise Data_Error with "malformed number in: " & Line;
   end Parse;

   function Load (Path : String; N_Input, N_Output : Positive) return Dataset is
      NF    : constant Positive := N_Input + N_Output;
      F     : File_Type;
      Count : Natural := 0;
   begin
      Open (F, In_File, Path);

      --  Pass 1: count data lines to size the allocation.
      while not End_Of_File (F) loop
         if Is_Data (To_String (Unbounded_IO.Get_Line (F))) then
            Count := Count + 1;
         end if;
      end loop;

      return D : Dataset do
         D.NI := N_Input;
         D.NO := N_Output;
         D.N  := Count;
         D.X  := new Real_Array (1 .. Count * N_Input);
         D.Y  := new Real_Array (1 .. Count * N_Output);

         Reset (F, In_File);

         --  Pass 2: parse each data line into its rows of X and Y.
         declare
            S : Natural := 0;
         begin
            while not End_Of_File (F) loop
               declare
                  Line : constant String :=
                    To_String (Unbounded_IO.Get_Line (F));
                  Vals : Real_Array (1 .. NF);
                  NT   : Natural;
               begin
                  if Is_Data (Line) then
                     Parse (Line, Vals, NT);
                     if NT /= NF then
                        raise Data_Error
                          with "expected" & Positive'Image (NF)
                               & " fields in: " & Line;
                     end if;
                     S := S + 1;
                     D.X ((S - 1) * N_Input + 1 .. S * N_Input) :=
                       Vals (1 .. N_Input);
                     D.Y ((S - 1) * N_Output + 1 .. S * N_Output) :=
                       Vals (N_Input + 1 .. NF);
                  end if;
               end;
            end loop;
         end;

         Close (F);
      end return;

   exception
      when others =>
         if Is_Open (F) then
            Close (F);
         end if;
         raise;
   end Load;

   function Samples  (D : Dataset) return Natural  is (D.N);
   function N_Input  (D : Dataset) return Positive is (D.NI);
   function N_Output (D : Dataset) return Positive is (D.NO);

   function Input (D : Dataset; I : Positive) return Real_Array is
     (D.X ((I - 1) * D.NI + 1 .. I * D.NI));

   function Target (D : Dataset; I : Positive) return Real_Array is
     (D.Y ((I - 1) * D.NO + 1 .. I * D.NO));

   overriding procedure Finalize (D : in out Dataset) is
   begin
      Free (D.X);
      Free (D.Y);
      D.N := 0;
   end Finalize;

   function Make_Split
     (D : Dataset'Class; Fraction : Real; G : in out SMBPANN.Rng.Generator)
      return Split
   is
      N : constant Natural := D.N;
   begin
      return S : Split do
         S.N     := N;
         S.Order := new Index_Array (1 .. N);
         for I in 1 .. N loop
            S.Order (I) := I;
         end loop;

         --  Fisher-Yates shuffle, so a fixed seed gives a fixed partition.
         for I in reverse 2 .. N loop
            declare
               J : constant Positive :=
                 1 + Natural (SMBPANN.Rng.Next (G)
                              mod Interfaces.Unsigned_32 (I));
               T : constant Positive := S.Order (I);
            begin
               S.Order (I) := S.Order (J);
               S.Order (J) := T;
            end;
         end loop;

         S.N_Train := Natural (Real'Floor (Fraction * Real (N)));
      end return;
   end Make_Split;

   function Train_Count (S : Split) return Natural is (S.N_Train);
   function Test_Count  (S : Split) return Natural is (S.N - S.N_Train);

   function Train_Index (S : Split; P : Positive) return Positive is
     (S.Order (P));

   function Test_Index (S : Split; P : Positive) return Positive is
     (S.Order (S.N_Train + P));

   overriding procedure Finalize (S : in out Split) is
   begin
      Free (S.Order);
      S.N := 0;
      S.N_Train := 0;
   end Finalize;

end SMBPANN.Data;
