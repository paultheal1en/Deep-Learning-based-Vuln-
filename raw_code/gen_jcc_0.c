static inline void gen_jcc(DisasContext *s, int b,

                           target_ulong val, target_ulong next_eip)

{

    int l1, l2;



    gen_update_cc_op(s);

    if (s->jmp_opt) {

        l1 = gen_new_label();

        gen_jcc1(s, b, l1);

        set_cc_op(s, CC_OP_DYNAMIC);

        

        gen_goto_tb(s, 0, next_eip);



        gen_set_label(l1);

        gen_goto_tb(s, 1, val);

        s->is_jmp = DISAS_TB_JUMP;

    } else {



        l1 = gen_new_label();

        l2 = gen_new_label();

        gen_jcc1(s, b, l1);

        set_cc_op(s, CC_OP_DYNAMIC);



        gen_jmp_im(next_eip);

        tcg_gen_br(l2);



        gen_set_label(l1);

        gen_jmp_im(val);

        gen_set_label(l2);

        gen_eob(s);

    }

}
