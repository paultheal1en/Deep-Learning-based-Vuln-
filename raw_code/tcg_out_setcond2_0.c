static void tcg_out_setcond2(TCGContext *s, TCGCond cond, TCGReg ret,

                             TCGReg al, TCGReg ah, TCGReg bl, TCGReg bh)

{

    TCGReg tmp0 = TCG_TMP0;

    TCGReg tmp1 = ret;



    assert(ret != TCG_TMP0);

    if (ret == ah || ret == bh) {

        assert(ret != TCG_TMP1);

        tmp1 = TCG_TMP1;

    }



    switch (cond) {

    case TCG_COND_EQ:

    case TCG_COND_NE:

        tmp1 = tcg_out_reduce_eq2(s, tmp0, tmp1, al, ah, bl, bh);

        tcg_out_setcond(s, cond, ret, tmp1, TCG_REG_ZERO);

        break;



    default:

        tcg_out_setcond(s, TCG_COND_EQ, tmp0, ah, bh);

        tcg_out_setcond(s, tcg_unsigned_cond(cond), tmp1, al, bl);

        tcg_out_opc_reg(s, OPC_AND, tmp1, tmp1, tmp0);

        tcg_out_setcond(s, tcg_high_cond(cond), tmp0, ah, bh);

        tcg_out_opc_reg(s, OPC_OR, ret, tmp1, tmp0);

        break;

    }

}
