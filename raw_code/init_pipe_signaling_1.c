static void init_pipe_signaling(PCIQXLDevice *d)

{

   if (pipe(d->pipe) < 0) {

       dprint(d, 1, "%s: pipe creation failed\n", __FUNCTION__);

       return;

   }

#ifdef CONFIG_IOTHREAD

   fcntl(d->pipe[0], F_SETFL, O_NONBLOCK);

#else

   fcntl(d->pipe[0], F_SETFL, O_NONBLOCK /* | O_ASYNC */);

#endif

   fcntl(d->pipe[1], F_SETFL, O_NONBLOCK);

   fcntl(d->pipe[0], F_SETOWN, getpid());



   d->main = pthread_self();

   qemu_set_fd_handler(d->pipe[0], pipe_read, NULL, d);

}
