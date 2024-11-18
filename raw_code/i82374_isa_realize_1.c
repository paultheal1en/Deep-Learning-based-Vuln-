static void i82374_isa_realize(DeviceState *dev, Error **errp)

{

    ISAi82374State *isa = I82374(dev);

    I82374State *s = &isa->state;

    PortioList *port_list = g_new(PortioList, 1);



    portio_list_init(port_list, OBJECT(isa), i82374_portio_list, s, "i82374");

    portio_list_add(port_list, isa_address_space_io(&isa->parent_obj),

                    isa->iobase);



    i82374_realize(s, errp);



    qdev_init_gpio_out(dev, &s->out, 1);

}
