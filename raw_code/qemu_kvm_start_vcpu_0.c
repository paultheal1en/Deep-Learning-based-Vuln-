static void qemu_kvm_start_vcpu(CPUState *env)

{

    env->thread = g_malloc0(sizeof(QemuThread));

    env->halt_cond = g_malloc0(sizeof(QemuCond));

    qemu_cond_init(env->halt_cond);

    qemu_thread_create(env->thread, qemu_kvm_cpu_thread_fn, env,

                       QEMU_THREAD_DETACHED);

    while (env->created == 0) {

        qemu_cond_wait(&qemu_cpu_cond, &qemu_global_mutex);

    }

}
