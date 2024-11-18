static int cpu_can_run(CPUState *env)

{

    if (env->stop)

        return 0;

    if (env->stopped)

        return 0;

    if (!vm_running)

        return 0;

    return 1;

}
