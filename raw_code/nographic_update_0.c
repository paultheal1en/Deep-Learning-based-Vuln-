static void nographic_update(void *opaque)

{

    uint64_t interval = GUI_REFRESH_INTERVAL;



    qemu_flush_coalesced_mmio_buffer();

    qemu_mod_timer(nographic_timer, interval + qemu_get_clock(rt_clock));

}
