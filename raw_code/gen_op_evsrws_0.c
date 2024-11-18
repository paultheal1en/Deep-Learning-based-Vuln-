static inline void gen_op_evsrws(TCGv_i32 ret, TCGv_i32 arg1, TCGv_i32 arg2)

{

    TCGv_i32 t0;

    int l1, l2;



    l1 = gen_new_label();

    l2 = gen_new_label();

    t0 = tcg_temp_local_new_i32();

    /* No error here: 6 bits are used */

    tcg_gen_andi_i32(t0, arg2, 0x3F);

    tcg_gen_brcondi_i32(TCG_COND_GE, t0, 32, l1);

    tcg_gen_sar_i32(ret, arg1, t0);

    tcg_gen_br(l2);

    gen_set_label(l1);

    tcg_gen_movi_i32(ret, 0);

    gen_set_label(l2);

    tcg_temp_free_i32(t0);

}
