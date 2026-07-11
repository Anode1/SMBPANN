--  smbpann-nets.ads -- a feed-forward network: flat weight buffers + forward pass.
--
--  The Java original modelled Network -> Layer -> Neuron -> Edge as an object
--  graph; here that collapses into flat weight arrays (layer L holds a
--  Units x Prev matrix, row-major: row receives, column sends -- the 1997
--  thesis convention). All buffers are allocated once in Create.
--
--  Network is a controlled type: its Finalize frees every buffer automatically
--  when the object leaves scope. Unlike the C version's net_free, you cannot
--  forget to call it -- the compiler inserts the cleanup. That is the Ada
--  answer to the manual-free burden.

with Ada.Finalization;
with SMBPANN.Rng;

package SMBPANN.Nets is

   type Index_Array is array (Positive range <>) of Positive;

   type Network is limited private;

   --  Build a network with the given layer sizes (Dims (1) = inputs,
   --  Dims (Dims'Last) = outputs). Allocates all buffers once.
   function Create (Dims : Index_Array) return Network
     with Pre => Dims'Length >= 2 and then Dims'First = 1;

   function N_Input  (Net : Network) return Positive;
   function N_Output (Net : Network) return Positive;
   function Num_Weights (Net : Network) return Natural;

   --  Initialize weights/biases uniformly in the thesis range
   --  [-2.4/fan_in, +2.4/fan_in], fan_in = the sending layer's width.
   procedure Initialize (Net : in out Network; G : in out SMBPANN.Rng.Generator);

   --  Forward pass: propagate X through every layer, return the output vector.
   function Forward (Net : in out Network; X : Real_Array) return Real_Array
     with Pre  => X'Length = N_Input (Net),
          Post => Forward'Result'Length = N_Output (Net);

private

   type Real_Array_Access is access Real_Array;

   type Layer_Rec is record
      Units : Positive := 1;
      Prev  : Natural  := 0;           --  0 for the input layer
      W     : Real_Array_Access;       --  Units * Prev, row-major (l >= 2)
      B     : Real_Array_Access;       --  Units                   (l >= 2)
      A     : Real_Array_Access;       --  Units activations       (l >= 1)
      Z     : Real_Array_Access;       --  Units pre-activations   (l >= 2)
   end record;

   type Layer_Vector is array (Positive range <>) of Layer_Rec;
   type Layer_Vector_Access is access Layer_Vector;

   type Network is new Ada.Finalization.Limited_Controlled with record
      N      : Natural := 0;           --  number of layers, input included
      Layers : Layer_Vector_Access;
   end record;

   overriding procedure Finalize (Net : in out Network);

end SMBPANN.Nets;
