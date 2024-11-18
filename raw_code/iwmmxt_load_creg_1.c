static inline TCGv iwmmxt_load_creg(int reg)

{

    TCGv var = new_tmp();

    tcg_gen_ld_i32(var, cpu_env, offsetof(CPUState, iwmmxt.cregs[reg]));

    return var;

}
