static void wdt_ib700_realize(DeviceState *dev, Error **errp)

{

    IB700State *s = IB700(dev);

    PortioList *port_list = g_new(PortioList, 1);



    ib700_debug("watchdog init\n");



    s->timer = timer_new_ns(QEMU_CLOCK_VIRTUAL, ib700_timer_expired, s);



    portio_list_init(port_list, OBJECT(s), wdt_portio_list, s, "ib700");

    portio_list_add(port_list, isa_address_space_io(&s->parent_obj), 0);

}
