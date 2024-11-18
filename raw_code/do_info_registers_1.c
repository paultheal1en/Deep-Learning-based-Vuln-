static void do_info_registers(Monitor *mon)

{

    CPUState *env;

    env = mon_get_cpu();

    if (!env)

        return;

#ifdef TARGET_I386

    cpu_dump_state(env, (FILE *)mon, monitor_fprintf,

                   X86_DUMP_FPU);

#else

    cpu_dump_state(env, (FILE *)mon, monitor_fprintf,

                   0);

#endif

}
