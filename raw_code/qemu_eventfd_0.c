int qemu_eventfd(int fds[2])

{

#ifdef CONFIG_EVENTFD

    int ret;



    ret = eventfd(0, 0);

    if (ret >= 0) {

        fds[0] = ret;

        qemu_set_cloexec(ret);

        if ((fds[1] = dup(ret)) == -1) {

            close(ret);

            return -1;

        }

        qemu_set_cloexec(fds[1]);

        return 0;

    }



    if (errno != ENOSYS) {

        return -1;

    }

#endif



    return qemu_pipe(fds);

}
