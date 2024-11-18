static void gen_vfp_msr(TCGv tmp)

{

    tcg_gen_mov_i32(cpu_F0s, tmp);

    dead_tmp(tmp);

}
