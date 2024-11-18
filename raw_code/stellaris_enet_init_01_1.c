void stellaris_enet_init(NICInfo *nd, uint32_t base, qemu_irq irq)

{

    stellaris_enet_state *s;

    int iomemtype;



    qemu_check_nic_model(nd, "stellaris");



    s = (stellaris_enet_state *)qemu_mallocz(sizeof(stellaris_enet_state));

    iomemtype = cpu_register_io_memory(0, stellaris_enet_readfn,

                                       stellaris_enet_writefn, s);

    cpu_register_physical_memory(base, 0x00001000, iomemtype);

    s->irq = irq;

    memcpy(s->macaddr, nd->macaddr, 6);



    if (nd->vlan) {

        s->vc = qemu_new_vlan_client(nd->vlan, nd->model, nd->name,

                                     stellaris_enet_receive, stellaris_enet_can_receive, s);

        qemu_format_nic_info_str(s->vc, s->macaddr);

    }



    stellaris_enet_reset(s);

    register_savevm("stellaris_enet", -1, 1,

                    stellaris_enet_save, stellaris_enet_load, s);

}
