--  smbpann-rng.ads -- deterministic 32-bit xorshift PRNG (Marsaglia 2003).
--
--  Reproducibility is a requirement: same seed, same run, every machine. A
--  modular type (Interfaces.Unsigned_32) makes the shift/xor arithmetic wrap by
--  definition -- no undefined behaviour, unlike C's signed shifts. Not crypto.

with Interfaces; use Interfaces;

package SMBPANN.Rng is

   type Generator is private;

   --  Any seed is accepted; 0 is remapped (xorshift cannot leave the zero
   --  state), so a 0 seed stays reproducible instead of degenerate.
   procedure Seed (G : out Generator; S : Unsigned_32);

   function Next (G : in out Generator) return Unsigned_32;

   --  Uniform real in [Lo, Hi).
   function Uniform (G : in out Generator; Lo, Hi : Real) return Real
     with Post => Uniform'Result >= Lo and then Uniform'Result < Hi;

private

   type Generator is record
      S : Unsigned_32 := 16#9E37_79B9#;
   end record;

end SMBPANN.Rng;
