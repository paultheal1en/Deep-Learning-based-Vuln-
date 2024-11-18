static void gen_addq_lo(DisasContext *s, TCGv val, int rlow)

{

    TCGv tmp;

    TCGv tmp2;



    /* Load value and extend to 64 bits.  */

    tmp = tcg_temp_new(TCG_TYPE_I64);

    tmp2 = load_reg(s, rlow);

    tcg_gen_extu_i32_i64(tmp, tmp2);

    dead_tmp(tmp2);

    tcg_gen_add_i64(val, val, tmp);

}
