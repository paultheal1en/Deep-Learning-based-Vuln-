static void gen_ld_asi(DisasContext *dc, TCGv dst, TCGv addr,

                       int insn, int size, int sign)

{

    TCGv_i32 r_asi, r_size, r_sign;



    r_asi = gen_get_asi(dc, insn);

    r_size = tcg_const_i32(size);

    r_sign = tcg_const_i32(sign);

#ifdef TARGET_SPARC64

    gen_helper_ld_asi(dst, cpu_env, addr, r_asi, r_size, r_sign);

#else

    {

        TCGv_i64 t64 = tcg_temp_new_i64();

        gen_helper_ld_asi(t64, cpu_env, addr, r_asi, r_size, r_sign);

        tcg_gen_trunc_i64_tl(dst, t64);

        tcg_temp_free_i64(t64);

    }

#endif

    tcg_temp_free_i32(r_sign);

    tcg_temp_free_i32(r_size);

    tcg_temp_free_i32(r_asi);

}
