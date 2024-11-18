static void gen_exception(int excp)

{

    TCGv tmp = new_tmp();

    tcg_gen_movi_i32(tmp, excp);

    gen_helper_exception(tmp);

    dead_tmp(tmp);

}
