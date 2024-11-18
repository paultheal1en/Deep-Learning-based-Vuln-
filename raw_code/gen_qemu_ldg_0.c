static always_inline void gen_qemu_ldg (TCGv t0, TCGv t1, int flags)

{

    TCGv tmp = tcg_temp_new(TCG_TYPE_I64);

    tcg_gen_qemu_ld64(tmp, t1, flags);

    tcg_gen_helper_1_1(helper_memory_to_g, t0, tmp);

    tcg_temp_free(tmp);

}
