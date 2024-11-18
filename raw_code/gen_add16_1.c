static void gen_add16(TCGv t0, TCGv t1)

{

    TCGv tmp = new_tmp();

    tcg_gen_xor_i32(tmp, t0, t1);

    tcg_gen_andi_i32(tmp, tmp, 0x8000);

    tcg_gen_andi_i32(t0, t0, ~0x8000);

    tcg_gen_andi_i32(t1, t1, ~0x8000);

    tcg_gen_add_i32(t0, t0, t1);

    tcg_gen_xor_i32(t0, t0, tmp);

    dead_tmp(tmp);

    dead_tmp(t1);

}
