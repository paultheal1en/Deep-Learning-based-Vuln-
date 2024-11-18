static void tcg_out_brcond2(TCGContext *s, TCGCond cond, TCGReg al, TCGReg ah,

                            TCGReg bl, TCGReg bh, int label_index)

{

    TCGCond b_cond = TCG_COND_NE;

    TCGReg tmp = TCG_TMP1;



    /* With branches, we emit between 4 and 9 insns with 2 or 3 branches.

       With setcond, we emit between 3 and 10 insns and only 1 branch,

       which ought to get better branch prediction.  */

     switch (cond) {

     case TCG_COND_EQ:

     case TCG_COND_NE:

        b_cond = cond;

        tmp = tcg_out_reduce_eq2(s, TCG_TMP0, TCG_TMP1, al, ah, bl, bh);

        break;



    default:

        /* Minimize code size by preferring a compare not requiring INV.  */

        if (mips_cmp_map[cond] & MIPS_CMP_INV) {

            cond = tcg_invert_cond(cond);

            b_cond = TCG_COND_EQ;

        }

        tcg_out_setcond2(s, cond, tmp, al, ah, bl, bh);

        break;

    }



    tcg_out_brcond(s, b_cond, tmp, TCG_REG_ZERO, label_index);

}
