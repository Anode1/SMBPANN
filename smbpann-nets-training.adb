--  smbpann-nets-training.adb -- backpropagation with momentum. See the spec.

with Ada.Unchecked_Deallocation;
with SMBPANN.Act;

package body SMBPANN.Nets.Training is

   procedure Free is
     new Ada.Unchecked_Deallocation (Real_Array, Real_Array_Access);
   procedure Free is
     new Ada.Unchecked_Deallocation (Scratch_Vector, Scratch_Access);

   function Create (Net : access Network; Rate, Momentum : Real) return Trainer
   is
   begin
      return T : Trainer (Net) do
         T.Rate     := Rate;
         T.Momentum := Momentum;
         T.S        := new Scratch_Vector (1 .. Net.N);
         for L in 2 .. Net.N loop
            declare
               Units : constant Positive := Net.Layers (L).Units;
               Prev  : constant Positive := Net.Layers (L).Prev;
            begin
               T.S (L).Beta := new Real_Array (1 .. Units);
               T.S (L).DW   := new Real_Array'(1 .. Units * Prev => 0.0);
               T.S (L).DB   := new Real_Array'(1 .. Units => 0.0);
            end;
         end loop;
      end return;
   end Create;

   function Learn (T : in out Trainer; X, D : Real_Array) return Real is
      Net : Network renames T.Net.all;
      NL  : constant Positive := Net.N;
      E   : Real := 0.0;
   begin
      --  Forward pass (updates the net's A/Z; the returned vector is unused
      --  because backprop reads the stored activations directly).
      declare
         R : constant Real_Array := Forward (Net, X);
         pragma Unreferenced (R);
      begin
         null;
      end;

      --  Output layer: beta = (d - a) * g'(z); accumulate squared error.
      declare
         OL : Layer_Rec   renames Net.Layers (NL);
         SB : Real_Array  renames T.S (NL).Beta.all;
      begin
         for I in 1 .. OL.Units loop
            declare
               Err : constant Real := D (I) - OL.A (I);
            begin
               E := E + 0.5 * Err * Err;
               SB (I) := Err * SMBPANN.Act.Sigmoid_Deriv (OL.A (I));
            end;
         end loop;
      end;

      --  Hidden layers, output-to-input. All betas use pre-update weights.
      for L in reverse 2 .. NL - 1 loop
         declare
            Cur   : Layer_Rec  renames Net.Layers (L);
            Nxt   : Layer_Rec  renames Net.Layers (L + 1);
            SB    : Real_Array renames T.S (L).Beta.all;
            NB    : Real_Array renames T.S (L + 1).Beta.all;
            NThis : constant Positive := Cur.Units;
         begin
            for I in 1 .. Cur.Units loop
               declare
                  Sum : Real := 0.0;
               begin
                  for K in 1 .. Nxt.Units loop
                     Sum := Sum + Nxt.W ((K - 1) * NThis + I) * NB (K);
                  end loop;
                  SB (I) := SMBPANN.Act.Sigmoid_Deriv (Cur.A (I)) * Sum;
               end;
            end loop;
         end;
      end loop;

      --  Apply updates: dw = rate*beta*a_prev + momentum*dw_prev; retain dw.
      for L in 2 .. NL loop
         declare
            Cur   : Layer_Rec  renames Net.Layers (L);
            Prev  : Real_Array renames Net.Layers (L - 1).A.all;
            NPrev : constant Positive := Cur.Prev;
            Bt    : Real_Array renames T.S (L).Beta.all;
            Dw    : Real_Array renames T.S (L).DW.all;
            Db    : Real_Array renames T.S (L).DB.all;
         begin
            for I in 1 .. Cur.Units loop
               declare
                  Bi : constant Real := Bt (I);
               begin
                  for J in 1 .. NPrev loop
                     declare
                        Idx  : constant Positive := (I - 1) * NPrev + J;
                        DeltaW : constant Real :=
                          T.Rate * Bi * Prev (J) + T.Momentum * Dw (Idx);
                     begin
                        Cur.W (Idx) := Cur.W (Idx) + DeltaW;
                        Dw (Idx)    := DeltaW;
                     end;
                  end loop;
                  declare
                     DeltaB : constant Real :=
                       T.Rate * Bi + T.Momentum * Db (I);
                  begin
                     Cur.B (I) := Cur.B (I) + DeltaB;
                     Db (I)    := DeltaB;
                  end;
               end;
            end loop;
         end;
      end loop;

      return E;
   end Learn;

   overriding procedure Finalize (T : in out Trainer) is
   begin
      if T.S /= null then
         for Sc of T.S.all loop
            Free (Sc.Beta);
            Free (Sc.DW);
            Free (Sc.DB);
         end loop;
         Free (T.S);
      end if;
   end Finalize;

end SMBPANN.Nets.Training;
