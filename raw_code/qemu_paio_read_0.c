int qemu_paio_read(struct qemu_paiocb *aiocb)

{

    return qemu_paio_submit(aiocb, QEMU_PAIO_READ);

}
