void qemu_bh_delete(QEMUBH *bh)

{

    qemu_free(bh);

}
