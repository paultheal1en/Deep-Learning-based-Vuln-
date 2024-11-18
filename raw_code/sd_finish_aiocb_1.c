static void coroutine_fn sd_finish_aiocb(SheepdogAIOCB *acb)

{

    qemu_coroutine_enter(acb->coroutine, NULL);

    qemu_aio_unref(acb);

}
