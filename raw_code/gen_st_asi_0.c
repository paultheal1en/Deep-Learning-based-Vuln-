static void gen_st_asi(DisasContext *dc, TCGv src, TCGv addr,

                       int insn, int size)

{

    TCGv_i32 r_asi, r_size;



    r_asi = gen_get_asi(dc, insn);

    r_size = tcg_const_i32(size);

#ifdef TARGET_SPARC64

    gen_helper_st_asi(cpu_env, addr, src, r_asi, r_size);

#else

    {

        TCGv_i64 t64 = tcg_temp_new_i64();

        tcg_gen_extu_tl_i64(t64, src);

        gen_helper_st_asi(cpu_env, addr, t64, r_asi, r_size);

        tcg_temp_free_i64(t64);

    }

#endif

    tcg_temp_free_i32(r_size);

    tcg_temp_free_i32(r_asi);

}
