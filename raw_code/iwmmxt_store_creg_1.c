static inline void iwmmxt_store_creg(int reg, TCGv var)

{

    tcg_gen_st_i32(var, cpu_env, offsetof(CPUState, iwmmxt.cregs[reg]));


}