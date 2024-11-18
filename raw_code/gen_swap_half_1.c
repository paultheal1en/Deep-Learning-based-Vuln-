static void gen_swap_half(TCGv var)

{

    TCGv tmp = new_tmp();

    tcg_gen_shri_i32(tmp, var, 16);

    tcg_gen_shli_i32(var, var, 16);

    tcg_gen_or_i32(var, var, tmp);

    dead_tmp(tmp);

}
