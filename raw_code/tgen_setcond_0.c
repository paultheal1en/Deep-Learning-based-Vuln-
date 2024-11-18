static void tgen_setcond(TCGContext *s, TCGType type, TCGCond cond,

                         TCGReg dest, TCGReg c1, TCGArg c2, int c2const)

{

    int cc;



    switch (cond) {

    case TCG_COND_GTU:

    case TCG_COND_GT:

    do_greater:

        /* The result of a compare has CC=2 for GT and CC=3 unused.

           ADD LOGICAL WITH CARRY considers (CC & 2) the carry bit.  */

        tgen_cmp(s, type, cond, c1, c2, c2const, true);

        tcg_out_movi(s, type, dest, 0);

        tcg_out_insn(s, RRE, ALCGR, dest, dest);

        return;



    case TCG_COND_GEU:

    do_geu:

        /* We need "real" carry semantics, so use SUBTRACT LOGICAL

           instead of COMPARE LOGICAL.  This needs an extra move.  */

        tcg_out_mov(s, type, TCG_TMP0, c1);

        if (c2const) {

            tcg_out_movi(s, TCG_TYPE_I64, dest, 0);

            if (type == TCG_TYPE_I32) {

                tcg_out_insn(s, RIL, SLFI, TCG_TMP0, c2);

            } else {

                tcg_out_insn(s, RIL, SLGFI, TCG_TMP0, c2);

            }

        } else {

            if (type == TCG_TYPE_I32) {

                tcg_out_insn(s, RR, SLR, TCG_TMP0, c2);

            } else {

                tcg_out_insn(s, RRE, SLGR, TCG_TMP0, c2);

            }

            tcg_out_movi(s, TCG_TYPE_I64, dest, 0);

        }

        tcg_out_insn(s, RRE, ALCGR, dest, dest);

        return;



    case TCG_COND_LEU:

    case TCG_COND_LTU:

    case TCG_COND_LT:

        /* Swap operands so that we can use GEU/GTU/GT.  */

        if (c2const) {

            tcg_out_movi(s, type, TCG_TMP0, c2);

            c2 = c1;

            c2const = 0;

            c1 = TCG_TMP0;

        } else {

            TCGReg t = c1;

            c1 = c2;

            c2 = t;

        }

        if (cond == TCG_COND_LEU) {

            goto do_geu;

        }

        cond = tcg_swap_cond(cond);

        goto do_greater;



    case TCG_COND_NE:

        /* X != 0 is X > 0.  */

        if (c2const && c2 == 0) {

            cond = TCG_COND_GTU;

            goto do_greater;

        }

        break;



    case TCG_COND_EQ:

        /* X == 0 is X <= 0 is 0 >= X.  */

        if (c2const && c2 == 0) {

            tcg_out_movi(s, TCG_TYPE_I64, TCG_TMP0, 0);

            c2 = c1;

            c2const = 0;

            c1 = TCG_TMP0;

            goto do_geu;

        }

        break;



    default:

        break;

    }



    cc = tgen_cmp(s, type, cond, c1, c2, c2const, false);

    if (facilities & FACILITY_LOAD_ON_COND) {

        /* Emit: d = 0, t = 1, d = (cc ? t : d).  */

        tcg_out_movi(s, TCG_TYPE_I64, dest, 0);

        tcg_out_movi(s, TCG_TYPE_I64, TCG_TMP0, 1);

        tcg_out_insn(s, RRF, LOCGR, dest, TCG_TMP0, cc);

    } else {

        /* Emit: d = 1; if (cc) goto over; d = 0; over:  */

        tcg_out_movi(s, type, dest, 1);

        tcg_out_insn(s, RI, BRC, cc, (4 + 4) >> 1);

        tcg_out_movi(s, type, dest, 0);

    }

}
