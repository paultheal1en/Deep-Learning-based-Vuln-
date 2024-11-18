static CharDriverState* create_eventfd_chr_device(void * opaque, EventNotifier *n,

                                                  int vector)

{

    /* create a event character device based on the passed eventfd */

    IVShmemState *s = opaque;

    CharDriverState * chr;

    int eventfd = event_notifier_get_fd(n);



    chr = qemu_chr_open_eventfd(eventfd);



    if (chr == NULL) {

        fprintf(stderr, "creating eventfd for eventfd %d failed\n", eventfd);

        exit(-1);

    }




    /* if MSI is supported we need multiple interrupts */

    if (ivshmem_has_feature(s, IVSHMEM_MSI)) {

        s->eventfd_table[vector].pdev = &s->dev;

        s->eventfd_table[vector].vector = vector;



        qemu_chr_add_handlers(chr, ivshmem_can_receive, fake_irqfd,

                      ivshmem_event, &s->eventfd_table[vector]);

    } else {

        qemu_chr_add_handlers(chr, ivshmem_can_receive, ivshmem_receive,

                      ivshmem_event, s);

    }



    return chr;



}