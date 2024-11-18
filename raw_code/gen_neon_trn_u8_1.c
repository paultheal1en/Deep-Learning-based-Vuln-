static void gen_neon_trn_u8(TCGv t0, TCGv t1)

{

    TCGv rd, tmp;



    rd = new_tmp();

    tmp = new_tmp();



    tcg_gen_shli_i32(rd, t0, 8);

    tcg_gen_andi_i32(rd, rd, 0xff00ff00);

    tcg_gen_andi_i32(tmp, t1, 0x00ff00ff);

    tcg_gen_or_i32(rd, rd, tmp);



    tcg_gen_shri_i32(t1, t1, 8);

    tcg_gen_andi_i32(t1, t1, 0x00ff00ff);

    tcg_gen_andi_i32(tmp, t0, 0xff00ff00);

    tcg_gen_or_i32(t1, t1, tmp);

    tcg_gen_mov_i32(t0, rd);



    dead_tmp(tmp);

    dead_tmp(rd);

}
