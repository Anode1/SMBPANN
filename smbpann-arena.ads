--  smbpann-arena.ads -- a marker/region (Mark-Release) storage pool.
--
--  A bump allocator over a fixed block of `Capacity` storage elements: `new`
--  advances a high-water mark; individual Deallocate is a no-op; Release rolls
--  the mark back to a saved point, reclaiming in one step everything allocated
--  since. This is the "manual heap with markers" discipline (Turbo Pascal's
--  Mark/Release), expressed as a real Ada storage pool -- so an access type
--  bound to it (`for P'Storage_Pool use A;`) allocates from the arena
--  transparently, and the numeric code keeps using plain array indexing.
--
--  For the genetic search this is the substrate the population carves from:
--    Save a mark -> build a generation of candidate networks -> Release ->
--  the whole generation is freed at once, no per-network teardown, no
--  fragmentation, and one bounded allocation (the arena) for the whole run.

with System.Storage_Pools;
with System.Storage_Elements;

package SMBPANN.Arena is

   use System.Storage_Elements;

   type Arena (Capacity : Storage_Count) is
     new System.Storage_Pools.Root_Storage_Pool with private;

   --  A saved high-water position, opaque to callers.
   type Mark is private;

   function Used        (A : Arena) return Storage_Count;
   function Capacity_Of (A : Arena) return Storage_Count;

   --  Remember the current top, to Release back to later.
   function Save (A : Arena) return Mark;

   --  Reclaim everything allocated since the mark. O(1); the arena's own
   --  memory is untouched. Any access value into the released region is
   --  dangling afterwards -- the one rule the caller must keep (stack discipline).
   procedure Release (A : in out Arena; To : Mark)
     with Post => Used (A) <= Capacity_Of (A);

   --  Reclaim everything.
   procedure Reset (A : in out Arena)
     with Post => Used (A) = 0;

private

   overriding procedure Allocate
     (A         : in out Arena;
      Address   : out System.Address;
      Size      : Storage_Count;
      Alignment : Storage_Count);

   overriding procedure Deallocate
     (A         : in out Arena;
      Address   : System.Address;
      Size      : Storage_Count;
      Alignment : Storage_Count) is null;   --  bump pool: single free is a no-op

   overriding function Storage_Size (A : Arena) return Storage_Count is
     (A.Capacity);

   type Arena (Capacity : Storage_Count) is
     new System.Storage_Pools.Root_Storage_Pool with record
      Store : Storage_Array (1 .. Capacity);
      Top   : Storage_Count := 0;          --  bytes handed out so far
   end record;

   type Mark is new Storage_Count;

end SMBPANN.Arena;
