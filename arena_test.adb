--  arena_test.adb -- exercises the Mark-Release arena storage pool.
--
--  Binds an access type to a local arena and checks that `new` allocates from
--  it, that Save/Release rolls the high-water mark back exactly, that Reset
--  empties it, and that over-capacity allocation raises Storage_Error.

with Ada.Text_IO;               use Ada.Text_IO;
with System.Storage_Elements;   use System.Storage_Elements;
with SMBPANN;                   use SMBPANN;
with SMBPANN.Arena;             use SMBPANN.Arena;

procedure Arena_Test is

   Pool : SMBPANN.Arena.Arena (Capacity => 64 * 1024);   --  64 KiB region

   type Buf is access Real_Array;
   for Buf'Storage_Pool use Pool;

   --  Pool is a storage pool: its Store component is raw backing memory, left
   --  intentionally uninitialized. GNAT's "may be referenced before it has a
   --  value" flow warning on Pool is therefore a false positive -- the pool's
   --  operations read only its Top counter (which defaults to 0).
   pragma Warnings (Off, Pool);

   Fail : Natural := 0;

   procedure Check (Cond : Boolean; Msg : String) is
   begin
      if Cond then
         Put_Line ("ok   " & Msg);
      else
         Fail := Fail + 1;
         Put_Line ("FAIL " & Msg);
      end if;
   end Check;

   U1 : Storage_Count;
   M  : Mark;
   X, Y : Buf;
begin
   Check (Used (Pool) = 0, "arena starts empty");

   X := new Real_Array (1 .. 100);
   Check (Used (Pool) > 0, "allocation advances the mark");
   X.all := (others => 1.5);
   Check (X (50) = 1.5, "allocated buffer is readable and writable");

   U1 := Used (Pool);
   M  := Save (Pool);
   Y  := new Real_Array (1 .. 200);
   Y.all := (others => 2.0);
   Check (Used (Pool) > U1, "second allocation grows past the saved mark");
   Check (X (50) = 1.5 and Y (200) = 2.0, "distinct buffers do not overlap");

   Release (Pool, M);
   Check (Used (Pool) = U1, "release rolls back exactly to the mark");

   Reset (Pool);
   Check (Used (Pool) = 0, "reset empties the arena");

   declare
      Over : Buf;
   begin
      Over := new Real_Array (1 .. 1_000_000);   --  ~4 MB >> 64 KiB
      Check (False, "over-capacity allocation should have raised");
      Over.all (1) := 0.0;   --  keep Over used
   exception
      when Storage_Error =>
         Check (True, "over-capacity allocation raises Storage_Error");
   end;

   New_Line;
   Put_Line (Integer'Image (7 - Fail) & " of 7 checks passed");
   if Fail > 0 then
      raise Program_Error with "arena_test failed";
   end if;
end Arena_Test;
