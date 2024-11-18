static inline void gen_op_evabs(TCGv_i32 ret, TCGv_i32 arg1)

{

    int l1 = gen_new_label();

    int l2 = gen_new_label();



    tcg_gen_brcondi_i32(TCG_COND_GE, arg1, 0, l1);

    tcg_gen_neg_i32(ret, arg1);

    tcg_gen_br(l2);

    gen_set_label(l1);

    tcg_gen_mov_i32(ret, arg1);

    gen_set_label(l2);

}
