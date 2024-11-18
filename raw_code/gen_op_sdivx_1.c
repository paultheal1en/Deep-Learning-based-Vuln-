static inline void gen_op_sdivx(TCGv dst, TCGv src1, TCGv src2)

{

    int l1, l2;



    l1 = gen_new_label();

    l2 = gen_new_label();

    tcg_gen_mov_tl(cpu_cc_src, src1);

    tcg_gen_mov_tl(cpu_cc_src2, src2);

    gen_trap_ifdivzero_tl(cpu_cc_src2);

    tcg_gen_brcondi_tl(TCG_COND_NE, cpu_cc_src, INT64_MIN, l1);

    tcg_gen_brcondi_tl(TCG_COND_NE, cpu_cc_src2, -1, l1);

    tcg_gen_movi_i64(dst, INT64_MIN);

    tcg_gen_br(l2);

    gen_set_label(l1);

    tcg_gen_div_i64(dst, cpu_cc_src, cpu_cc_src2);

    gen_set_label(l2);

}
