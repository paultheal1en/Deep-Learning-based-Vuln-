static void gen_neon_dup_high16(TCGv var)

{

    TCGv tmp = new_tmp();

    tcg_gen_andi_i32(var, var, 0xffff0000);

    tcg_gen_shri_i32(tmp, var, 16);

    tcg_gen_or_i32(var, var, tmp);

    dead_tmp(tmp);

}
