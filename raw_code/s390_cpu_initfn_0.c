static void s390_cpu_initfn(Object *obj)

{

    CPUState *cs = CPU(obj);

    S390CPU *cpu = S390_CPU(obj);

    CPUS390XState *env = &cpu->env;

    static bool inited;

    static int cpu_num = 0;

#if !defined(CONFIG_USER_ONLY)

    struct tm tm;

#endif



    cs->env_ptr = env;

    cpu_exec_init(env);

#if !defined(CONFIG_USER_ONLY)

    qemu_register_reset(s390_cpu_machine_reset_cb, cpu);

    qemu_get_timedate(&tm, 0);

    env->tod_offset = TOD_UNIX_EPOCH +

                      (time2tod(mktimegm(&tm)) * 1000000000ULL);

    env->tod_basetime = 0;

    env->tod_timer = timer_new_ns(QEMU_CLOCK_VIRTUAL, s390x_tod_timer, cpu);

    env->cpu_timer = timer_new_ns(QEMU_CLOCK_VIRTUAL, s390x_cpu_timer, cpu);

    s390_cpu_set_state(CPU_STATE_STOPPED, cpu);

#endif

    env->cpu_num = cpu_num++;

    env->ext_index = -1;



    if (tcg_enabled() && !inited) {

        inited = true;

        s390x_translate_init();

    }

}
