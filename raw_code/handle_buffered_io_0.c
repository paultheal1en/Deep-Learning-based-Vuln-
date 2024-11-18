static void handle_buffered_io(void *opaque)

{

    XenIOState *state = opaque;



    if (handle_buffered_iopage(state)) {

        timer_mod(state->buffered_io_timer,

                BUFFER_IO_MAX_DELAY + qemu_clock_get_ms(QEMU_CLOCK_REALTIME));

    } else {

        timer_del(state->buffered_io_timer);

        xc_evtchn_unmask(state->xce_handle, state->bufioreq_local_port);

    }

}
