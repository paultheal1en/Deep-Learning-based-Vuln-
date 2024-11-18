void run_on_cpu(CPUState *env, void (*func)(void *data), void *data)

{

    struct qemu_work_item wi;



    if (qemu_cpu_self(env)) {

        func(data);

        return;

    }



    wi.func = func;

    wi.data = data;

    if (!env->queued_work_first)

        env->queued_work_first = &wi;

    else

        env->queued_work_last->next = &wi;

    env->queued_work_last = &wi;

    wi.next = NULL;

    wi.done = false;



    qemu_cpu_kick(env);

    while (!wi.done) {

        CPUState *self_env = cpu_single_env;



        qemu_cond_wait(&qemu_work_cond, &qemu_global_mutex);

        cpu_single_env = self_env;

    }

}
