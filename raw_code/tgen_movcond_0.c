static void tgen_movcond(TCGContext *s, TCGType type, TCGCond c, TCGReg dest,

                         TCGReg c1, TCGArg c2, int c2const, TCGReg r3)

{

    int cc;

    if (facilities & FACILITY_LOAD_ON_COND) {

        cc = tgen_cmp(s, type, c, c1, c2, c2const, false);

        tcg_out_insn(s, RRF, LOCGR, dest, r3, cc);

    } else {

        c = tcg_invert_cond(c);

        cc = tgen_cmp(s, type, c, c1, c2, c2const, false);



        /* Emit: if (cc) goto over; dest = r3; over:  */

        tcg_out_insn(s, RI, BRC, cc, (4 + 4) >> 1);

        tcg_out_insn(s, RRE, LGR, dest, r3);

    }

}
