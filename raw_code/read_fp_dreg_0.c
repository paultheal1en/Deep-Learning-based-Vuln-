static TCGv_i64 read_fp_dreg(DisasContext *s, int reg)

{

    TCGv_i64 v = tcg_temp_new_i64();



    tcg_gen_ld_i64(v, cpu_env, fp_reg_offset(reg, MO_64));

    return v;

}
