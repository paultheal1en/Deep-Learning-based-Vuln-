static void gen_neon_dup_low16(TCGv var)

{

    TCGv tmp = new_tmp();

    tcg_gen_ext16u_i32(var, var);

    tcg_gen_shli_i32(tmp, var, 16);

    tcg_gen_or_i32(var, var, tmp);

    dead_tmp(tmp);

}
