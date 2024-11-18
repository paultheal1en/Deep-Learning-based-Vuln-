static void nic_cleanup(NetClientState *nc)

{

    dp8393xState *s = qemu_get_nic_opaque(nc);



    memory_region_del_subregion(s->address_space, &s->mmio);

    memory_region_destroy(&s->mmio);



    timer_del(s->watchdog);

    timer_free(s->watchdog);



    g_free(s);

}
