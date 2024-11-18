static void tcg_out_movcond32(TCGContext *s, TCGCond cond, TCGArg dest,

                              TCGArg c1, TCGArg c2, int const_c2,

                              TCGArg v1)

{

    tcg_out_cmp(s, c1, c2, const_c2, 0);

    if (have_cmov) {

        tcg_out_modrm(s, OPC_CMOVCC | tcg_cond_to_jcc[cond], dest, v1);

    } else {

        int over = gen_new_label();

        tcg_out_jxx(s, tcg_cond_to_jcc[tcg_invert_cond(cond)], over, 1);

        tcg_out_mov(s, TCG_TYPE_I32, dest, v1);

        tcg_out_label(s, over, s->code_ptr);

    }

}
