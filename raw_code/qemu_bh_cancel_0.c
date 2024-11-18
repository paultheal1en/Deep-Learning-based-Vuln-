void qemu_bh_cancel(QEMUBH *bh)

{

    bh->scheduled = 0;

}
