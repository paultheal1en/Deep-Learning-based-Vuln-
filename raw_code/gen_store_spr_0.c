static always_inline void gen_store_spr(int reg, TCGv t)

{

    tcg_gen_st_tl(t, cpu_env, offsetof(CPUState, spr[reg]));

}
