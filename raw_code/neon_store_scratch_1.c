static void neon_store_scratch(int scratch, TCGv var)

{

    tcg_gen_st_i32(var, cpu_env, offsetof(CPUARMState, vfp.scratch[scratch]));

    dead_tmp(var);

}
