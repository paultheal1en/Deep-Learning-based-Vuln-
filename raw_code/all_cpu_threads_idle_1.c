static bool all_cpu_threads_idle(void)

{

    CPUState *env;



    for (env = first_cpu; env != NULL; env = env->next_cpu) {

        if (!cpu_thread_is_idle(env)) {

            return false;

        }

    }

    return true;

}
