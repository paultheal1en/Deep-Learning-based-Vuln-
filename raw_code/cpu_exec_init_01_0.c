void cpu_exec_init(CPUState *env)

{

    CPUState **penv;

    int cpu_index;



#if defined(CONFIG_USER_ONLY)

    cpu_list_lock();

#endif

    env->next_cpu = NULL;

    penv = &first_cpu;

    cpu_index = 0;

    while (*penv != NULL) {

        penv = &(*penv)->next_cpu;

        cpu_index++;

    }

    env->cpu_index = cpu_index;

    env->numa_node = 0;

    TAILQ_INIT(&env->breakpoints);

    TAILQ_INIT(&env->watchpoints);

    *penv = env;

#if defined(CONFIG_USER_ONLY)

    cpu_list_unlock();

#endif

#if defined(CPU_SAVE_VERSION) && !defined(CONFIG_USER_ONLY)

    vmstate_register(cpu_index, &vmstate_cpu_common, env);

    register_savevm("cpu", cpu_index, CPU_SAVE_VERSION,

                    cpu_save, cpu_load, env);

#endif

}
