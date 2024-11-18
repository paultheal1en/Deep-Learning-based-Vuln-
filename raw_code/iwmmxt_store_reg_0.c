static inline void iwmmxt_store_reg(TCGv var, int reg)

{

    tcg_gen_st_i64(var, cpu_env, offsetof(CPUState, iwmmxt.regs[reg]));

}
