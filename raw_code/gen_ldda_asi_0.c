static void gen_ldda_asi(DisasContext *dc, TCGv hi, TCGv addr,

                         int insn, int rd)

{

    TCGv_i32 r_asi, r_rd;



    r_asi = gen_get_asi(dc, insn);

    r_rd = tcg_const_i32(rd);

    gen_helper_ldda_asi(cpu_env, addr, r_asi, r_rd);

    tcg_temp_free_i32(r_rd);

    tcg_temp_free_i32(r_asi);

}
