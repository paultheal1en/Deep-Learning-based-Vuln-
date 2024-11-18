CPUX86State *cpu_x86_init(const char *cpu_model)

{

    CPUX86State *env;

    static int inited;



    env = qemu_mallocz(sizeof(CPUX86State));

    cpu_exec_init(env);

    env->cpu_model_str = cpu_model;



    /* init various static tables */

    if (!inited) {

        inited = 1;

        optimize_flags_init();

#ifndef CONFIG_USER_ONLY

        prev_debug_excp_handler =

            cpu_set_debug_excp_handler(breakpoint_handler);

#endif

    }

    if (cpu_x86_register(env, cpu_model) < 0) {

        cpu_x86_close(env);

        return NULL;

    }

    mce_init(env);

    cpu_reset(env);

#ifdef CONFIG_KQEMU

    kqemu_init(env);

#endif



    qemu_init_vcpu(env);



    return env;

}
