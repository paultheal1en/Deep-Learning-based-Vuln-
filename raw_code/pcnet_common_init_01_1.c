static void pcnet_common_init(PCNetState *d, NICInfo *nd)

{

    d->poll_timer = qemu_new_timer(vm_clock, pcnet_poll_timer, d);



    d->nd = nd;



    if (nd && nd->vlan) {

        d->vc = qemu_new_vlan_client(nd->vlan, nd->model, nd->name,

                                     pcnet_receive, pcnet_can_receive, d);



        qemu_format_nic_info_str(d->vc, d->nd->macaddr);

    } else {

        d->vc = NULL;

    }

    pcnet_h_reset(d);

    register_savevm("pcnet", -1, 2, pcnet_save, pcnet_load, d);

}
