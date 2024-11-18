static void qemu_tcg_init_vcpu(void *_env)

{

    CPUState *env = _env;

    /* share a single thread for all cpus with TCG */

    if (!tcg_cpu_thread) {

        env->thread = qemu_mallocz(sizeof(QemuThread));

        env->halt_cond = qemu_mallocz(sizeof(QemuCond));

        qemu_cond_init(env->halt_cond);

        qemu_thread_create(env->thread, qemu_tcg_cpu_thread_fn, env);

        while (env->created == 0)

            qemu_cond_timedwait(&qemu_cpu_cond, &qemu_global_mutex, 100);

        tcg_cpu_thread = env->thread;

        tcg_halt_cond = env->halt_cond;

    } else {

        env->thread = tcg_cpu_thread;

        env->halt_cond = tcg_halt_cond;

    }

}
