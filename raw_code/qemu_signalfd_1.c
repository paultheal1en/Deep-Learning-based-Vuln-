int qemu_signalfd(const sigset_t *mask)

{

#if defined(CONFIG_signalfd)

    int ret;



    ret = syscall(SYS_signalfd, -1, mask, _NSIG / 8);

    if (ret != -1)

        return ret;

#endif



    return qemu_signalfd_compat(mask);

}
