--  smbpann-act.adb -- activation functions. See smbpann-act.ads.

with Ada.Numerics.Generic_Elementary_Functions;

package body SMBPANN.Act is

   package Math is new Ada.Numerics.Generic_Elementary_Functions (Real);

   function Sigmoid (Z : Real) return Real is
   begin
      --  Numerically stable form: never Exp() a large positive argument.
      if Z >= 0.0 then
         return 1.0 / (1.0 + Math.Exp (-Z));
      else
         declare
            E : constant Real := Math.Exp (Z);
         begin
            return E / (1.0 + E);
         end;
      end if;
   end Sigmoid;

   function Sigmoid_Deriv (A : Real) return Real is
   begin
      return A * (1.0 - A);
   end Sigmoid_Deriv;

end SMBPANN.Act;
