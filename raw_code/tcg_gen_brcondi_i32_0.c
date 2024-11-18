void tcg_gen_brcondi_i32(TCGCond cond, TCGv_i32 arg1, int32_t arg2, int label)

{

    TCGv_i32 t0 = tcg_const_i32(arg2);

    tcg_gen_brcond_i32(cond, arg1, t0, label);

    tcg_temp_free_i32(t0);

}
