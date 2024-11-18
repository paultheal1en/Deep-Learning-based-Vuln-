static void gen_interrupt(DisasContext *s, int intno,

                          target_ulong cur_eip, target_ulong next_eip)

{

    gen_update_cc_op(s);

    gen_jmp_im(cur_eip);

    gen_helper_raise_interrupt(cpu_env, tcg_const_i32(intno),

                               tcg_const_i32(next_eip - cur_eip));

    s->is_jmp = DISAS_TB_JUMP;

}
