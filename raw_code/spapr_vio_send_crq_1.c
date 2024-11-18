int spapr_vio_send_crq(VIOsPAPRDevice *dev, uint8_t *crq)

{

    int rc;

    uint8_t byte;



    if (!dev->crq.qsize) {

        fprintf(stderr, "spapr_vio_send_creq on uninitialized queue\n");

        return -1;

    }



    /* Maybe do a fast path for KVM just writing to the pages */

    rc = spapr_tce_dma_read(dev, dev->crq.qladdr + dev->crq.qnext, &byte, 1);

    if (rc) {

        return rc;

    }

    if (byte != 0) {

        return 1;

    }



    rc = spapr_tce_dma_write(dev, dev->crq.qladdr + dev->crq.qnext + 8,

                             &crq[8], 8);

    if (rc) {

        return rc;

    }



    kvmppc_eieio();



    rc = spapr_tce_dma_write(dev, dev->crq.qladdr + dev->crq.qnext, crq, 8);

    if (rc) {

        return rc;

    }



    dev->crq.qnext = (dev->crq.qnext + 16) % dev->crq.qsize;



    if (dev->signal_state & 1) {

        qemu_irq_pulse(dev->qirq);

    }



    return 0;

}
