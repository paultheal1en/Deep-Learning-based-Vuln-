static int cpu_has_work(CPUState *env)

{

    if (env->stop)

        return 1;

    if (env->stopped)

        return 0;

    if (!env->halted)

        return 1;

    if (qemu_cpu_has_work(env))

        return 1;

    return 0;

}
