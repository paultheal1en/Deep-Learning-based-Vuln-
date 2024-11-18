static void gen_ldf_asi(DisasContext *dc, TCGv addr,

                        int insn, int size, int rd)

{

    TCGv_i32 r_asi, r_size, r_rd;



    r_asi = gen_get_asi(dc, insn);

    r_size = tcg_const_i32(size);

    r_rd = tcg_const_i32(rd);

    gen_helper_ldf_asi(cpu_env, addr, r_asi, r_size, r_rd);

    tcg_temp_free_i32(r_rd);

    tcg_temp_free_i32(r_size);

    tcg_temp_free_i32(r_asi);

}
