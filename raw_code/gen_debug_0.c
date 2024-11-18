static void gen_debug(DisasContext *s, target_ulong cur_eip)

{

    gen_update_cc_op(s);

    gen_jmp_im(cur_eip);

    gen_helper_debug(cpu_env);

    s->is_jmp = DISAS_TB_JUMP;

}
