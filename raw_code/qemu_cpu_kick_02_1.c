void qemu_cpu_kick(void *_env)

{

    CPUState *env = _env;

    qemu_cond_broadcast(env->halt_cond);

    qemu_thread_signal(env->thread, SIG_IPI);

}
