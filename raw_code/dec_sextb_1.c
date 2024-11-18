static void dec_sextb(DisasContext *dc)

{

    LOG_DIS("sextb r%d, r%d\n", dc->r2, dc->r0);



    if (!(dc->env->features & LM32_FEATURE_SIGN_EXTEND)) {

        cpu_abort(dc->env, "hardware sign extender is not available\n");

    }



    tcg_gen_ext8s_tl(cpu_R[dc->r2], cpu_R[dc->r0]);

}
