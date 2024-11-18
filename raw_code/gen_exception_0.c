static void gen_exception(DisasContext *s, int trapno, target_ulong cur_eip)

{

    gen_update_cc_op(s);

    gen_jmp_im(cur_eip);

    gen_helper_raise_exception(cpu_env, tcg_const_i32(trapno));

    s->is_jmp = DISAS_TB_JUMP;

}
