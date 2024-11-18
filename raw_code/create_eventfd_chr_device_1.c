static CharDriverState* create_eventfd_chr_device(IVShmemState *s,

                                                  EventNotifier *n,

                                                  int vector)

{

    /* create a event character device based on the passed eventfd */

    PCIDevice *pdev = PCI_DEVICE(s);

    int eventfd = event_notifier_get_fd(n);

    CharDriverState *chr;



    s->msi_vectors[vector].pdev = pdev;



    chr = qemu_chr_open_eventfd(eventfd);



    if (chr == NULL) {

        error_report("creating chardriver for eventfd %d failed", eventfd);

        return NULL;

    }

    qemu_chr_fe_claim_no_fail(chr);



    /* if MSI is supported we need multiple interrupts */

    if (ivshmem_has_feature(s, IVSHMEM_MSI)) {

        s->msi_vectors[vector].pdev = PCI_DEVICE(s);



        qemu_chr_add_handlers(chr, ivshmem_can_receive, fake_irqfd,

                      ivshmem_event, &s->msi_vectors[vector]);

    } else {

        qemu_chr_add_handlers(chr, ivshmem_can_receive, ivshmem_receive,

                      ivshmem_event, s);

    }



    return chr;



}
