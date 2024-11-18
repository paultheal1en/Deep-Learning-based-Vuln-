void serial_realize_core(SerialState *s, Error **errp)

{

    if (!qemu_chr_fe_backend_connected(&s->chr)) {

        error_setg(errp, "Can't create serial device, empty char device");

        return;

    }



    s->modem_status_poll = timer_new_ns(QEMU_CLOCK_VIRTUAL, (QEMUTimerCB *) serial_update_msl, s);



    s->fifo_timeout_timer = timer_new_ns(QEMU_CLOCK_VIRTUAL, (QEMUTimerCB *) fifo_timeout_int, s);

    qemu_register_reset(serial_reset, s);



    qemu_chr_fe_set_handlers(&s->chr, serial_can_receive1, serial_receive1,

                             serial_event, NULL, s, NULL, true);

    fifo8_create(&s->recv_fifo, UART_FIFO_LENGTH);

    fifo8_create(&s->xmit_fifo, UART_FIFO_LENGTH);

    serial_reset(s);

}
