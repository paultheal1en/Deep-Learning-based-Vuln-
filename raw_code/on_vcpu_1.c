static void on_vcpu(CPUState *env, void (*func)(void *data), void *data)

{

    if (env == cpu_single_env) {

        func(data);

        return;

    }

    abort();

}
