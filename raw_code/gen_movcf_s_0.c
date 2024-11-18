static inline void gen_movcf_s (int fs, int fd, int cc, int tf)

{

    uint32_t ccbit;

    int cond;

    TCGv r_tmp1 = tcg_temp_local_new(TCG_TYPE_I32);

    TCGv fp0 = tcg_temp_local_new(TCG_TYPE_I32);

    TCGv fp1 = tcg_temp_local_new(TCG_TYPE_I32);

    int l1 = gen_new_label();



    if (cc)

        ccbit = 1 << (24 + cc);

    else

        ccbit = 1 << 23;



    if (tf)

        cond = TCG_COND_EQ;

    else

        cond = TCG_COND_NE;



    gen_load_fpr32(fp0, fs);

    gen_load_fpr32(fp1, fd);

    tcg_gen_andi_i32(r_tmp1, fpu_fcr31, ccbit);

    tcg_gen_brcondi_i32(cond, r_tmp1, 0, l1);

    tcg_gen_mov_i32(fp1, fp0);

    tcg_temp_free(fp0);

    gen_set_label(l1);

    tcg_temp_free(r_tmp1);

    gen_store_fpr32(fp1, fd);

    tcg_temp_free(fp1);

}
