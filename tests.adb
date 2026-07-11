--  tests.adb -- in-process unit tests for the SMBPANN engine.
--
--  The Ada analogue of the C reference's tests.c: linear, inline, one check per
--  assertion. Covers rng, act, nets, the XOR backprop regression, and the arena.
--  `make test` builds and runs it; a non-zero exit marks failure.

with Ada.Text_IO;               use Ada.Text_IO;
with Ada.Command_Line;          use Ada.Command_Line;
with Ada.Directories;
with Interfaces;                use type Interfaces.Unsigned_32;
with System.Storage_Elements;   use System.Storage_Elements;
with SMBPANN;                   use SMBPANN;
with SMBPANN.Rng;
with SMBPANN.Act;
with SMBPANN.Nets;              use SMBPANN.Nets;
with SMBPANN.Nets.Training;     use SMBPANN.Nets.Training;
with SMBPANN.Arena;             use SMBPANN.Arena;
with SMBPANN.Data;              use SMBPANN.Data;

procedure Tests is

   Total, Fail : Natural := 0;

   procedure Check (Cond : Boolean; Msg : String) is
   begin
      Total := Total + 1;
      if Cond then
         Put_Line ("ok   " & Msg);
      else
         Fail := Fail + 1;
         Put_Line ("FAIL " & Msg);
      end if;
   end Check;

begin
   --  rng: reproducible, in range, and 0-seed remapped (not stuck at zero)
   declare
      G, H : SMBPANN.Rng.Generator;
      Z    : SMBPANN.Rng.Generator;
      U    : Real;
   begin
      SMBPANN.Rng.Seed (G, 42);
      SMBPANN.Rng.Seed (H, 42);
      Check (SMBPANN.Rng.Next (G) = SMBPANN.Rng.Next (H),
             "rng: same seed gives the same sequence");
      U := SMBPANN.Rng.Uniform (G, -1.0, 1.0);
      Check (U >= -1.0 and then U < 1.0, "rng: uniform stays in [lo, hi)");
      SMBPANN.Rng.Seed (Z, 0);
      Check (SMBPANN.Rng.Next (Z) /= 0, "rng: seed 0 is remapped, not degenerate");
   end;

   --  act: sigmoid pins, saturation, and the a*(1-a) derivative
   Check (abs (SMBPANN.Act.Sigmoid (0.0) - 0.5) < 1.0e-6,
          "act: sigmoid(0) = 0.5");
   Check (SMBPANN.Act.Sigmoid (100.0) <= 1.0
          and then SMBPANN.Act.Sigmoid (-100.0) >= 0.0,
          "act: sigmoid saturates within [0, 1]");
   Check (abs (SMBPANN.Act.Sigmoid_Deriv (0.5) - 0.25) < 1.0e-6,
          "act: sigmoid' at a=0.5 is 0.25");

   --  nets: shape queries, deterministic forward pass, output range
   declare
      Net    : Network := Create ((2, 3, 1));
      G      : SMBPANN.Rng.Generator;
      X      : constant Real_Array := (0.5, 0.25);
      Y1, Y2 : Real_Array (1 .. 1);
   begin
      Check (N_Input (Net) = 2 and then N_Output (Net) = 1,
             "nets: input and output widths");
      Check (Num_Weights (Net) = 9, "nets: 2-3-1 has 9 weights");
      SMBPANN.Rng.Seed (G, 7);
      Initialize (Net, G);
      Y1 := Forward (Net, X);
      Y2 := Forward (Net, X);
      Check (Y1 (1) = Y2 (1), "nets: forward pass is deterministic");
      Check (Y1 (1) > 0.0 and then Y1 (1) < 1.0, "nets: sigmoid output in (0,1)");
   end;

   --  training: the XOR regression -- backprop through a hidden layer learns
   --  the linearly-non-separable function below an error threshold
   declare
      type Sample is record
         X : Real_Array (1 .. 2);
         D : Real_Array (1 .. 1);
      end record;
      Data : constant array (1 .. 4) of Sample :=
        (((0.0, 0.0), (1 => 0.0)),
         ((0.0, 1.0), (1 => 1.0)),
         ((1.0, 0.0), (1 => 1.0)),
         ((1.0, 1.0), (1 => 0.0)));
      Net : aliased Network := Create ((2, 4, 1));
      G   : SMBPANN.Rng.Generator;
      E   : Real := 0.0;
      OK  : Boolean := True;
   begin
      SMBPANN.Rng.Seed (G, 1);
      Initialize (Net, G);
      declare
         T : Trainer := Create (Net'Access, 0.5, 0.9);
      begin
         for Epoch in 1 .. 20_000 loop
            E := 0.0;
            for S of Data loop
               E := E + Learn (T, S.X, S.D);
            end loop;
         end loop;
      end;
      Check (E < 0.02, "training: XOR converges to sse < 0.02");
      for S of Data loop
         declare
            Y : constant Real_Array := Forward (Net, S.X);
         begin
            if (if Y (1) > 0.5 then 1.0 else 0.0) /= S.D (1) then
               OK := False;
            end if;
         end;
      end loop;
      Check (OK, "training: XOR classifies all four patterns correctly");
   end;

   --  arena: allocate via the pool, roll back to a mark, reset, and overflow
   declare
      Pool : SMBPANN.Arena.Arena (Capacity => 64 * 1024);
      type Buf is access Real_Array;
      for Buf'Storage_Pool use Pool;
      --  Pool is raw backing memory; GNAT's "referenced before value" flow
      --  warning on it is a false positive (only its Top counter is read).
      pragma Warnings (Off, Pool);
      U1   : Storage_Count;
      M    : Mark;
      X, Y : Buf;
   begin
      Check (Used (Pool) = 0, "arena: starts empty");
      X := new Real_Array (1 .. 100);
      X.all := (others => 1.5);
      U1 := Used (Pool);
      M  := Save (Pool);
      Y  := new Real_Array (1 .. 200);
      Y.all := (others => 2.0);
      Check (Used (Pool) > U1, "arena: allocation advances the mark");
      Check (X (50) = 1.5 and then Y (200) = 2.0, "arena: buffers do not overlap");
      Release (Pool, M);
      Check (Used (Pool) = U1, "arena: release rolls back to the mark");
      Reset (Pool);
      Check (Used (Pool) = 0, "arena: reset empties the arena");
      declare
         Over : Buf;
      begin
         Over := new Real_Array (1 .. 1_000_000);
         Check (False, "arena: over-capacity allocation should raise");
         Over.all (1) := 0.0;
      exception
         when Storage_Error =>
            Check (True, "arena: over-capacity allocation raises Storage_Error");
      end;
   end;

   --  data: load a plain-text file (comment + integer tokens), then split it
   declare
      Path : constant String := "smbpann_data_test.tmp";
      G    : SMBPANN.Rng.Generator;
   begin
      declare
         TF : File_Type;
      begin
         Create (TF, Out_File, Path);
         Put_Line (TF, "# xor sample set");
         Put_Line (TF, "0 0 0");
         Put_Line (TF, "0 1 1");
         Put_Line (TF, "1 0 1");
         Put_Line (TF, "1 1 0");
         Close (TF);
      end;

      declare
         DS : constant Dataset := Load (Path, 2, 1);
      begin
         Check (Samples (DS) = 4, "data: loads all rows, skips the comment");
         Check (N_Input (DS) = 2 and then N_Output (DS) = 1,
                "data: input/output widths");
         Check (Input (DS, 2) = (0.0, 1.0) and then Target (DS, 2) = (1 => 1.0),
                "data: row values parsed (integer tokens accepted)");

         SMBPANN.Rng.Seed (G, 5);
         declare
            Sp   : constant Split := Make_Split (DS, 0.75, G);
            Seen : array (1 .. 4) of Boolean := (others => False);
         begin
            Check (Train_Count (Sp) = 3 and then Test_Count (Sp) = 1,
                   "data: 75% split -> 3 train, 1 test");
            for P in 1 .. Train_Count (Sp) loop
               Seen (Train_Index (Sp, P)) := True;
            end loop;
            for P in 1 .. Test_Count (Sp) loop
               Seen (Test_Index (Sp, P)) := True;
            end loop;
            Check (Seen = (True, True, True, True),
                   "data: split covers every sample exactly once");
         end;
      end;

      Ada.Directories.Delete_File (Path);
   end;

   New_Line;
   Put_Line (Integer'Image (Total - Fail) & " of" & Integer'Image (Total)
             & " checks passed");
   if Fail > 0 then
      Set_Exit_Status (Failure);
   end if;
end Tests;
