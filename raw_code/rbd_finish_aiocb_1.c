static void rbd_finish_aiocb(rbd_completion_t c, RADOSCB *rcb)

{

    int ret;

    rcb->ret = rbd_aio_get_return_value(c);

    rbd_aio_release(c);

    ret = qemu_rbd_send_pipe(rcb->s, rcb);

    if (ret < 0) {

        error_report("failed writing to acb->s->fds");

        g_free(rcb);

    }

}
