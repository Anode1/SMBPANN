--  smbpann-rng.adb -- xorshift PRNG. See smbpann-rng.ads.

package body SMBPANN.Rng is

   procedure Seed (G : out Generator; S : Unsigned_32) is
   begin
      if S = 0 then
         G.S := 16#9E37_79B9#;
      else
         G.S := S;
      end if;
   end Seed;

   function Next (G : in out Generator) return Unsigned_32 is
      X : Unsigned_32 := G.S;
   begin
      X := X xor Shift_Left  (X, 13);
      X := X xor Shift_Right (X, 17);
      X := X xor Shift_Left  (X, 5);
      G.S := X;
      return X;
   end Next;

   function Uniform (G : in out Generator; Lo, Hi : Real) return Real is
      --  Full 2**32 divisor maps the word into [0, 1).
      U : constant Real := Real (Next (G)) / 4_294_967_296.0;
   begin
      return Lo + U * (Hi - Lo);
   end Uniform;

end SMBPANN.Rng;
