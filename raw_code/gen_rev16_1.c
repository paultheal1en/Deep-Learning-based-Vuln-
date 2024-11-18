static void gen_rev16(TCGv var)

{

    TCGv tmp = new_tmp();

    tcg_gen_shri_i32(tmp, var, 8);

    tcg_gen_andi_i32(tmp, tmp, 0x00ff00ff);

    tcg_gen_shli_i32(var, var, 8);

    tcg_gen_andi_i32(var, var, 0xff00ff00);

    tcg_gen_or_i32(var, var, tmp);

    dead_tmp(tmp);

}
