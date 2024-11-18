bool qemu_signalfd_available(void)

{

#ifdef CONFIG_SIGNALFD

    errno = 0;

    syscall(SYS_signalfd, -1, NULL, _NSIG / 8);

    return errno != ENOSYS;

#else

    return false;

#endif

}
