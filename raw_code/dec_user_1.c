static void dec_user(DisasContext *dc)

{

    LOG_DIS("user");



    cpu_abort(dc->env, "user insn undefined\n");

}
