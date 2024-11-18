static void qemu_laio_completion_cb(void *opaque)

{

    struct qemu_laio_state *s = opaque;



    while (1) {

        struct io_event events[MAX_EVENTS];

        uint64_t val;

        ssize_t ret;

        struct timespec ts = { 0 };

        int nevents, i;



        do {

            ret = read(s->efd, &val, sizeof(val));

        } while (ret == 1 && errno == EINTR);



        if (ret == -1 && errno == EAGAIN)

            break;



        if (ret != 8)

            break;



        do {

            nevents = io_getevents(s->ctx, val, MAX_EVENTS, events, &ts);

        } while (nevents == -EINTR);



        for (i = 0; i < nevents; i++) {

            struct iocb *iocb = events[i].obj;

            struct qemu_laiocb *laiocb =

                    container_of(iocb, struct qemu_laiocb, iocb);



            laiocb->ret = io_event_ret(&events[i]);

            qemu_laio_enqueue_completed(s, laiocb);

        }

    }

}
