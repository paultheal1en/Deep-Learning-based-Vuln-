static always_inline void gen_qemu_stg (TCGv t0, TCGv t1, int flags)

{

    TCGv tmp = tcg_temp_new(TCG_TYPE_I64);

    tcg_gen_helper_1_1(helper_g_to_memory, tmp, t0);

    tcg_gen_qemu_st64(tmp, t1, flags);

    tcg_temp_free(tmp);

}
