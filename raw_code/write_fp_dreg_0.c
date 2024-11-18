static void write_fp_dreg(DisasContext *s, int reg, TCGv_i64 v)

{

    TCGv_i64 tcg_zero = tcg_const_i64(0);



    tcg_gen_st_i64(v, cpu_env, fp_reg_offset(reg, MO_64));

    tcg_gen_st_i64(tcg_zero, cpu_env, fp_reg_hi_offset(reg));

    tcg_temp_free_i64(tcg_zero);

}
