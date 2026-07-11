--  smbpann-arena.adb -- the Mark-Release bump allocator. See smbpann-arena.ads.

package body SMBPANN.Arena is

   overriding procedure Allocate
     (A         : in out Arena;
      Address   : out System.Address;
      Size      : Storage_Count;
      Alignment : Storage_Count)
   is
      Start : Storage_Count;
   begin
      --  Round the current top up to the requested alignment, then check the
      --  request fits before handing out the address and bumping the top.
      Start := ((A.Top + Alignment - 1) / Alignment) * Alignment;
      if Start + Size > A.Capacity then
         raise Storage_Error with "SMBPANN arena exhausted";
      end if;
      Address := A.Store (Start + 1)'Address;
      A.Top   := Start + Size;
   end Allocate;

   function Used        (A : Arena) return Storage_Count is (A.Top);
   function Capacity_Of (A : Arena) return Storage_Count is (A.Capacity);

   function Save (A : Arena) return Mark is (Mark (A.Top));

   procedure Release (A : in out Arena; To : Mark) is
   begin
      A.Top := Storage_Count (To);
   end Release;

   procedure Reset (A : in out Arena) is
   begin
      A.Top := 0;
   end Reset;

end SMBPANN.Arena;
