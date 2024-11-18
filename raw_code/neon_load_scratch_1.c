static TCGv neon_load_scratch(int scratch)

{

    TCGv tmp = new_tmp();

    tcg_gen_ld_i32(tmp, cpu_env, offsetof(CPUARMState, vfp.scratch[scratch]));

    return tmp;

}
