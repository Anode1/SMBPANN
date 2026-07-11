--  smbpann-nets.adb -- feed-forward network and forward pass. See the spec.
--
--  Allocation lives only in Create; Finalize releases it; Forward and
--  Initialize touch no allocator (the C version's "hot path allocates nothing"
--  invariant, here enforced by there being no `new` outside Create).

with Ada.Unchecked_Deallocation;
with SMBPANN.Act;

package body SMBPANN.Nets is

   procedure Free is
     new Ada.Unchecked_Deallocation (Real_Array, Real_Array_Access);
   procedure Free is
     new Ada.Unchecked_Deallocation (Layer_Vector, Layer_Vector_Access);

   function N_Input  (Net : Network) return Positive is (Net.Layers (1).Units);
   function N_Output (Net : Network) return Positive is (Net.Layers (Net.N).Units);

   function Num_Weights (Net : Network) return Natural is
      Total : Natural := 0;
   begin
      for L in 2 .. Net.N loop
         Total := Total + Net.Layers (L).Units * Net.Layers (L).Prev;
      end loop;
      return Total;
   end Num_Weights;

   function Create (Dims : Index_Array) return Network is
   begin
      return Net : Network do
         Net.N := Dims'Length;
         Net.Layers := new Layer_Vector (1 .. Dims'Length);
         for L in 1 .. Dims'Length loop
            declare
               Units : constant Positive := Dims (L);
               Prev  : constant Natural  := (if L = 1 then 0 else Dims (L - 1));
            begin
               Net.Layers (L).Units := Units;
               Net.Layers (L).Prev  := Prev;
               Net.Layers (L).A     := new Real_Array (1 .. Units);
               if Prev > 0 then
                  Net.Layers (L).W := new Real_Array (1 .. Units * Prev);
                  Net.Layers (L).B := new Real_Array (1 .. Units);
                  Net.Layers (L).Z := new Real_Array (1 .. Units);
               end if;
            end;
         end loop;
      end return;
   end Create;

   procedure Initialize (Net : in out Network; G : in out SMBPANN.Rng.Generator)
   is
   begin
      for L in 2 .. Net.N loop
         declare
            Cur   : Layer_Rec renames Net.Layers (L);
            Bound : constant Real := 2.4 / Real (Cur.Prev);
         begin
            for I in Cur.W'Range loop
               Cur.W (I) := SMBPANN.Rng.Uniform (G, -Bound, Bound);
            end loop;
            for I in Cur.B'Range loop
               Cur.B (I) := SMBPANN.Rng.Uniform (G, -Bound, Bound);
            end loop;
         end;
      end loop;
   end Initialize;

   function Forward (Net : in out Network; X : Real_Array) return Real_Array is
   begin
      Net.Layers (1).A.all := X;
      for L in 2 .. Net.N loop
         declare
            Cur   : Layer_Rec renames Net.Layers (L);
            Prev  : Real_Array renames Net.Layers (L - 1).A.all;
            NPrev : constant Positive := Cur.Prev;
         begin
            for I in 1 .. Cur.Units loop
               declare
                  S : Real := Cur.B (I);
               begin
                  for J in 1 .. NPrev loop
                     S := S + Cur.W ((I - 1) * NPrev + J) * Prev (J);
                  end loop;
                  Cur.Z (I) := S;
                  Cur.A (I) := SMBPANN.Act.Sigmoid (S);
               end;
            end loop;
         end;
      end loop;
      return Net.Layers (Net.N).A.all;
   end Forward;

   overriding procedure Finalize (Net : in out Network) is
   begin
      if Net.Layers /= null then
         for L of Net.Layers.all loop
            Free (L.W);
            Free (L.B);
            Free (L.A);
            Free (L.Z);
         end loop;
         Free (Net.Layers);
      end if;
      Net.N := 0;
   end Finalize;

end SMBPANN.Nets;
