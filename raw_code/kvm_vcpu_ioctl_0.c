int kvm_vcpu_ioctl(CPUState *env, int type, ...)

{

    int ret;

    void *arg;

    va_list ap;



    va_start(ap, type);

    arg = va_arg(ap, void *);

    va_end(ap);



    ret = ioctl(env->kvm_fd, type, arg);

    if (ret == -1)

        ret = -errno;



    return ret;

}
