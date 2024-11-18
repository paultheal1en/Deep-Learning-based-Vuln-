static void nic_cleanup(VLANClientState *nc)

{

    dp8393xState *s = DO_UPCAST(NICState, nc, nc)->opaque;



    cpu_unregister_io_memory(s->mmio_index);



    qemu_del_timer(s->watchdog);

    qemu_free_timer(s->watchdog);



    g_free(s);

}
