void qemu_init_vcpu(void *_env)

{

    CPUState *env = _env;



    if (kvm_enabled())

        kvm_init_vcpu(env);

    env->nr_cores = smp_cores;

    env->nr_threads = smp_threads;

    return;

}
