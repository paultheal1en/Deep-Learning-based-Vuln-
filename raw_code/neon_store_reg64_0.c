static inline void neon_store_reg64(TCGv var, int reg)

{

    tcg_gen_st_i64(var, cpu_env, vfp_reg_offset(1, reg));

}
