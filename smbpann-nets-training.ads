--  smbpann-nets-training.ads -- backpropagation: the generalized delta rule.
--
--  A child of SMBPANN.Nets so it can see the private Layer representation, the
--  Ada idiom for "these two concepts share internals" without exposing them to
--  the world. The mathematics is the 1997 thesis:
--     output:  beta = (d - a) * g'(z)
--     hidden:  beta = g'(z) * sum_k w[l+1][k][i] * beta[l+1][k]
--     update:  dw = rate*beta*a_prev + momentum*dw_prev ;  w := w + dw
--
--  Trainer is controlled: its scratch (error signals + previous deltas for
--  momentum) is allocated once in Create and freed automatically by Finalize.

with Ada.Finalization;

package SMBPANN.Nets.Training is

   type Trainer (Net : access Network) is limited private;

   function Create (Net : access Network; Rate, Momentum : Real) return Trainer;

   --  One online backprop step on example (X, D): forward, back-propagate, and
   --  update every weight/bias in place. Returns the pre-update error
   --  E = 0.5 * sum (D - A)**2.
   function Learn (T : in out Trainer; X, D : Real_Array) return Real
     with Pre => X'Length = N_Input (T.Net.all)
                 and then D'Length = N_Output (T.Net.all);

private

   type Scratch_Rec is record
      Beta : Real_Array_Access;   --  Units error signals
      DW   : Real_Array_Access;   --  previous weight deltas (W's shape)
      DB   : Real_Array_Access;   --  previous bias deltas   (B's shape)
   end record;

   type Scratch_Vector is array (Positive range <>) of Scratch_Rec;
   type Scratch_Access is access Scratch_Vector;

   type Trainer (Net : access Network) is
     new Ada.Finalization.Limited_Controlled with record
      Rate     : Real := 0.0;
      Momentum : Real := 0.0;
      S        : Scratch_Access;
   end record;

   overriding procedure Finalize (T : in out Trainer);

end SMBPANN.Nets.Training;
