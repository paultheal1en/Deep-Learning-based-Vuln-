int qemu_cpu_self(void *_env)

{

    CPUState *env = _env;

    QemuThread this;



    qemu_thread_self(&this);



    return qemu_thread_equal(&this, env->thread);

}
