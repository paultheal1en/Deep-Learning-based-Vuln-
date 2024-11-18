void isa_ne2000_init(int base, qemu_irq irq, NICInfo *nd)

{

    NE2000State *s;



    qemu_check_nic_model(nd, "ne2k_isa");



    s = qemu_mallocz(sizeof(NE2000State));



    register_ioport_write(base, 16, 1, ne2000_ioport_write, s);

    register_ioport_read(base, 16, 1, ne2000_ioport_read, s);



    register_ioport_write(base + 0x10, 1, 1, ne2000_asic_ioport_write, s);

    register_ioport_read(base + 0x10, 1, 1, ne2000_asic_ioport_read, s);

    register_ioport_write(base + 0x10, 2, 2, ne2000_asic_ioport_write, s);

    register_ioport_read(base + 0x10, 2, 2, ne2000_asic_ioport_read, s);



    register_ioport_write(base + 0x1f, 1, 1, ne2000_reset_ioport_write, s);

    register_ioport_read(base + 0x1f, 1, 1, ne2000_reset_ioport_read, s);

    s->irq = irq;

    memcpy(s->macaddr, nd->macaddr, 6);



    ne2000_reset(s);



    s->vc = qemu_new_vlan_client(nd->vlan, nd->model, nd->name,

                                 ne2000_receive, ne2000_can_receive, s);



    qemu_format_nic_info_str(s->vc, s->macaddr);



    register_savevm("ne2000", -1, 2, ne2000_save, ne2000_load, s);

}
