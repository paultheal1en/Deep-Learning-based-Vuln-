int qemu_bh_poll(void)

{

    return aio_bh_poll(qemu_aio_context);

}
