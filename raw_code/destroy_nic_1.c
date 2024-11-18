void destroy_nic(dev_match_fn *match_fn, void *arg)

{

    int i;

    NICInfo *nic;



    for (i = 0; i < MAX_NICS; i++) {

        nic = &nd_table[i];

        if (nic->used) {

            if (nic->private && match_fn(nic->private, arg)) {

                if (nic->vlan) {

                    VLANClientState *vc;

                    vc = qemu_find_vlan_client(nic->vlan, nic->private);

                    if (vc)

                        qemu_del_vlan_client(vc);

                }

                net_client_uninit(nic);

            }

        }

    }

}
