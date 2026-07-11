--  smbpann.ads -- root package: the scalar type shared across the engine.
--
--  SMBPANN in Ada, an experiment in expressing the AIS coding discipline where
--  the compiler enforces it by construction rather than by convention. `Real`
--  is the one numeric type (Float = 32-bit) the whole engine uses; one line
--  changes it to Long_Float, exactly as `smb_real` does in the C version.

package SMBPANN is
   pragma Pure;

   type Real is new Float;

   type Real_Array is array (Positive range <>) of Real;

end SMBPANN;
