static void kvm_start_vcpu(CPUState *env)

{

    env->thread = qemu_mallocz(sizeof(QemuThread));

    env->halt_cond = qemu_mallocz(sizeof(QemuCond));

    qemu_cond_init(env->halt_cond);

    qemu_thread_create(env->thread, kvm_cpu_thread_fn, env);

    while (env->created == 0)

        qemu_cond_timedwait(&qemu_cpu_cond, &qemu_global_mutex, 100);

}
