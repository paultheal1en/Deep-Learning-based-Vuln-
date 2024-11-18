static void ioq_init(LaioQueue *io_q)

{

    QSIMPLEQ_INIT(&io_q->pending);

    io_q->plugged = 0;

    io_q->n = 0;

    io_q->blocked = false;

}
