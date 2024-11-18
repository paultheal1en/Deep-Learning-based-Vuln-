static void handle_rev16(DisasContext *s, unsigned int sf,

                         unsigned int rn, unsigned int rd)

{

    TCGv_i64 tcg_rd = cpu_reg(s, rd);

    TCGv_i64 tcg_tmp = tcg_temp_new_i64();

    TCGv_i64 tcg_rn = read_cpu_reg(s, rn, sf);

    TCGv_i64 mask = tcg_const_i64(sf ? 0x00ff00ff00ff00ffull : 0x00ff00ff);



    tcg_gen_shri_i64(tcg_tmp, tcg_rn, 8);

    tcg_gen_and_i64(tcg_rd, tcg_rn, mask);

    tcg_gen_and_i64(tcg_tmp, tcg_tmp, mask);

    tcg_gen_shli_i64(tcg_rd, tcg_rd, 8);

    tcg_gen_or_i64(tcg_rd, tcg_rd, tcg_tmp);




    tcg_temp_free_i64(tcg_tmp);

}