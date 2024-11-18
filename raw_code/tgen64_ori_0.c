static void tgen64_ori(TCGContext *s, TCGReg dest, tcg_target_ulong val)

{

    static const S390Opcode oi_insns[4] = {

        RI_OILL, RI_OILH, RI_OIHL, RI_OIHH

    };

    static const S390Opcode nif_insns[2] = {

        RIL_OILF, RIL_OIHF

    };



    int i;



    /* Look for no-op.  */

    if (val == 0) {

        return;

    }



    if (facilities & FACILITY_EXT_IMM) {

        /* Try all 32-bit insns that can perform it in one go.  */

        for (i = 0; i < 4; i++) {

            tcg_target_ulong mask = (0xffffull << i*16);

            if ((val & mask) != 0 && (val & ~mask) == 0) {

                tcg_out_insn_RI(s, oi_insns[i], dest, val >> i*16);

                return;

            }

        }



        /* Try all 48-bit insns that can perform it in one go.  */

        for (i = 0; i < 2; i++) {

            tcg_target_ulong mask = (0xffffffffull << i*32);

            if ((val & mask) != 0 && (val & ~mask) == 0) {

                tcg_out_insn_RIL(s, nif_insns[i], dest, val >> i*32);

                return;

            }

        }



        /* Perform the OR via sequential modifications to the high and

           low parts.  Do this via recursion to handle 16-bit vs 32-bit

           masks in each half.  */

        tgen64_ori(s, dest, val & 0x00000000ffffffffull);

        tgen64_ori(s, dest, val & 0xffffffff00000000ull);

    } else {

        /* With no extended-immediate facility, we don't need to be so

           clever.  Just iterate over the insns and mask in the constant.  */

        for (i = 0; i < 4; i++) {

            tcg_target_ulong mask = (0xffffull << i*16);

            if ((val & mask) != 0) {

                tcg_out_insn_RI(s, oi_insns[i], dest, val >> i*16);

            }

        }

    }

}
