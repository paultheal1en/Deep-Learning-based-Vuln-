static inline void gen_goto_tb(DisasContext *s, int tb_num, target_ulong eip)

{

    target_ulong pc = s->cs_base + eip;



    if (use_goto_tb(s, pc))  {

        /* jump to same page: we can use a direct jump */

        tcg_gen_goto_tb(tb_num);

        gen_jmp_im(eip);

        tcg_gen_exit_tb((uintptr_t)s->tb + tb_num);

    } else {

        /* jump to another page: currently not optimized */

        gen_jmp_im(eip);

        gen_eob(s);

    }

}
