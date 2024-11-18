static int tcg_match_ori(TCGType type, tcg_target_long val)

{

    if (facilities & FACILITY_EXT_IMM) {

        if (type == TCG_TYPE_I32) {

            /* All 32-bit ORs can be performed with 1 48-bit insn.  */

            return 1;

        }

    }



    /* Look for negative values.  These are best to load with LGHI.  */

    if (val < 0) {

        if (val == (int16_t)val) {

            return 0;

        }

        if (facilities & FACILITY_EXT_IMM) {

            if (val == (int32_t)val) {

                return 0;

            }

        }

    }



    return 1;

}
