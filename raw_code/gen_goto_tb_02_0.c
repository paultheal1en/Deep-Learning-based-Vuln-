static inline void gen_goto_tb(DisasContext *s, int n, uint64_t dest)

{

    TranslationBlock *tb;



    tb = s->tb;

    if (use_goto_tb(s, n, dest)) {

        tcg_gen_goto_tb(n);

        gen_a64_set_pc_im(dest);

        tcg_gen_exit_tb((intptr_t)tb + n);

        s->is_jmp = DISAS_TB_JUMP;

    } else {

        gen_a64_set_pc_im(dest);

        if (s->singlestep_enabled) {

            gen_exception_internal(EXCP_DEBUG);

        }

        tcg_gen_exit_tb(0);

        s->is_jmp = DISAS_JUMP;

    }

}
