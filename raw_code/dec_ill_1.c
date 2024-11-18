static void dec_ill(DisasContext *dc)

{

    cpu_abort(dc->env, "unknown opcode 0x%02x\n", dc->opcode);

}
