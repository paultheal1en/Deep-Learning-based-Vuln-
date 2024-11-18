int qemu_pipe(int pipefd[2])

{

    int ret;



#ifdef CONFIG_PIPE2

    ret = pipe2(pipefd, O_CLOEXEC);

#else

    ret = pipe(pipefd);

    if (ret == 0) {

        qemu_set_cloexec(pipefd[0]);

        qemu_set_cloexec(pipefd[1]);

    }

#endif



    return ret;

}
