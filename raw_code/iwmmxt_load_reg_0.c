static inline void iwmmxt_load_reg(TCGv var, int reg)

{

    tcg_gen_ld_i64(var, cpu_env, offsetof(CPUState, iwmmxt.regs[reg]));

}
