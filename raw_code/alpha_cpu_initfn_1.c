static void alpha_cpu_initfn(Object *obj)

{

    CPUState *cs = CPU(obj);

    AlphaCPU *cpu = ALPHA_CPU(obj);

    CPUAlphaState *env = &cpu->env;



    cs->env_ptr = env;

    cpu_exec_init(cs, &error_abort);

    tlb_flush(cs, 1);



    alpha_translate_init();



#if defined(CONFIG_USER_ONLY)

    env->ps = PS_USER_MODE;

    cpu_alpha_store_fpcr(env, (FPCR_INVD | FPCR_DZED | FPCR_OVFD

                               | FPCR_UNFD | FPCR_INED | FPCR_DNOD

                               | FPCR_DYN_NORMAL));

#endif

    env->lock_addr = -1;

    env->fen = 1;

}
