static void qxl_send_events(PCIQXLDevice *d, uint32_t events)

{

    uint32_t old_pending;

    uint32_t le_events = cpu_to_le32(events);



    trace_qxl_send_events(d->id, events);

    assert(qemu_spice_display_is_running(&d->ssd));

    old_pending = __sync_fetch_and_or(&d->ram->int_pending, le_events);

    if ((old_pending & le_events) == le_events) {

        return;

    }

    if (qemu_thread_is_self(&d->main)) {

        qxl_update_irq(d);

    } else {

        if (write(d->pipe[1], d, 1) != 1) {

            dprint(d, 1, "%s: write to pipe failed\n", __func__);

        }

    }

}
