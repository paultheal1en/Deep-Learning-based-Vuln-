static inline void gen_movcf_d (DisasContext *ctx, int fs, int fd, int cc, int tf)

{

    int cond;

    TCGv_i32 t0 = tcg_temp_new_i32();

    TCGv_i64 fp0;

    int l1 = gen_new_label();



    if (tf)

        cond = TCG_COND_EQ;

    else

        cond = TCG_COND_NE;



    tcg_gen_andi_i32(t0, fpu_fcr31, 1 << get_fp_bit(cc));

    tcg_gen_brcondi_i32(cond, t0, 0, l1);

    tcg_temp_free_i32(t0);

    fp0 = tcg_temp_new_i64();

    gen_load_fpr64(ctx, fp0, fs);

    gen_store_fpr64(ctx, fp0, fd);

    tcg_temp_free_i64(fp0);

    gen_set_label(l1);

}
