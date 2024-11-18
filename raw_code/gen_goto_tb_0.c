static inline void gen_goto_tb(DisasContext *s, int n, target_ulong dest)

{

    if (use_goto_tb(s, dest)) {

        tcg_gen_goto_tb(n);

        gen_set_pc_im(s, dest);

        tcg_gen_exit_tb((uintptr_t)s->tb + n);

    } else {

        TCGv addr = tcg_temp_new();



        gen_set_pc_im(s, dest);

        tcg_gen_extu_i32_tl(addr, cpu_R[15]);

        tcg_gen_lookup_and_goto_ptr(addr);

        tcg_temp_free(addr);

    }

}
