static void serial_init_core(SerialState *s)

{

    if (!s->chr) {

        fprintf(stderr, "Can't create serial device, empty char device\n");

	exit(1);

    }



    s->modem_status_poll = qemu_new_timer(vm_clock, (QEMUTimerCB *) serial_update_msl, s);



    s->fifo_timeout_timer = qemu_new_timer(vm_clock, (QEMUTimerCB *) fifo_timeout_int, s);

    s->transmit_timer = qemu_new_timer(vm_clock, (QEMUTimerCB *) serial_xmit, s);



    qemu_register_reset(serial_reset, s);

    serial_reset(s);



    qemu_chr_add_handlers(s->chr, serial_can_receive1, serial_receive1,

                          serial_event, s);

}
