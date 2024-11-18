static always_inline void gen_load_spr(TCGv t, int reg)

{

    tcg_gen_ld_tl(t, cpu_env, offsetof(CPUState, spr[reg]));

}
