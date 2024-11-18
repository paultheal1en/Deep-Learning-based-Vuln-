static void bmdma_irq(void *opaque, int n, int level)

{

    BMDMAState *bm = opaque;



    if (!level) {

        /* pass through lower */

        qemu_set_irq(bm->irq, level);

        return;

    }



    if (bm) {

        bm->status |= BM_STATUS_INT;

    }



    /* trigger the real irq */

    qemu_set_irq(bm->irq, level);

}
