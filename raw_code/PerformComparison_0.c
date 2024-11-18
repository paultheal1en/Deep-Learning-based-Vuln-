static unsigned int PerformComparison(const unsigned int opcode)

{

   FPA11 *fpa11 = GET_FPA11();

   unsigned int Fn, Fm;

   floatx80 rFn, rFm;

   int e_flag = opcode & 0x400000;	/* 1 if CxFE */

   int n_flag = opcode & 0x200000;	/* 1 if CNxx */

   unsigned int flags = 0;



   //printk("PerformComparison(0x%08x)\n",opcode);



   Fn = getFn(opcode);

   Fm = getFm(opcode);



   /* Check for unordered condition and convert all operands to 80-bit

      format.

      ?? Might be some mileage in avoiding this conversion if possible.

      Eg, if both operands are 32-bit, detect this and do a 32-bit

      comparison (cheaper than an 80-bit one).  */

   switch (fpa11->fType[Fn])

   {

      case typeSingle:

        //printk("single.\n");

	if (float32_is_nan(fpa11->fpreg[Fn].fSingle))

	   goto unordered;

        rFn = float32_to_floatx80(fpa11->fpreg[Fn].fSingle, &fpa11->fp_status);

      break;



      case typeDouble:

        //printk("double.\n");

	if (float64_is_nan(fpa11->fpreg[Fn].fDouble))

	   goto unordered;

        rFn = float64_to_floatx80(fpa11->fpreg[Fn].fDouble, &fpa11->fp_status);

      break;



      case typeExtended:

        //printk("extended.\n");

	if (floatx80_is_nan(fpa11->fpreg[Fn].fExtended))

	   goto unordered;

        rFn = fpa11->fpreg[Fn].fExtended;

      break;



      default: return 0;

   }



   if (CONSTANT_FM(opcode))

   {

     //printk("Fm is a constant: #%d.\n",Fm);

     rFm = getExtendedConstant(Fm);

     if (floatx80_is_nan(rFm))

        goto unordered;

   }

   else

   {

     //printk("Fm = r%d which contains a ",Fm);

      switch (fpa11->fType[Fm])

      {

         case typeSingle:

           //printk("single.\n");

	   if (float32_is_nan(fpa11->fpreg[Fm].fSingle))

	      goto unordered;

           rFm = float32_to_floatx80(fpa11->fpreg[Fm].fSingle, &fpa11->fp_status);

         break;



         case typeDouble:

           //printk("double.\n");

	   if (float64_is_nan(fpa11->fpreg[Fm].fDouble))

	      goto unordered;

           rFm = float64_to_floatx80(fpa11->fpreg[Fm].fDouble, &fpa11->fp_status);

         break;



         case typeExtended:

           //printk("extended.\n");

	   if (floatx80_is_nan(fpa11->fpreg[Fm].fExtended))

	      goto unordered;

           rFm = fpa11->fpreg[Fm].fExtended;

         break;



         default: return 0;

      }

   }



   if (n_flag)

   {

      rFm.high ^= 0x8000;

   }



   return PerformComparisonOperation(rFn,rFm);



 unordered:

   /* ?? The FPA data sheet is pretty vague about this, in particular

      about whether the non-E comparisons can ever raise exceptions.

      This implementation is based on a combination of what it says in

      the data sheet, observation of how the Acorn emulator actually

      behaves (and how programs expect it to) and guesswork.  */

   flags |= CC_OVERFLOW;

   flags &= ~(CC_ZERO | CC_NEGATIVE);



   if (BIT_AC & readFPSR()) flags |= CC_CARRY;



   if (e_flag) float_raise(float_flag_invalid, &fpa11->fp_status);



   writeConditionCodes(flags);

   return 1;

}
