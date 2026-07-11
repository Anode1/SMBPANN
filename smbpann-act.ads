--  smbpann-act.ads -- activation functions and their derivatives.
--
--  The 1997 thesis derives backpropagation with the logistic sigmoid, whose
--  derivative has the closed form g'(z) = a*(1-a) in terms of the activation a
--  itself -- so the derivative takes the already-computed activation, not z.
--  The Post contract states the sigmoid's range; -gnata turns it into a check.

package SMBPANN.Act is
   pragma Pure;

   function Sigmoid (Z : Real) return Real
     with Post => Sigmoid'Result >= 0.0 and then Sigmoid'Result <= 1.0;

   function Sigmoid_Deriv (A : Real) return Real;

end SMBPANN.Act;
