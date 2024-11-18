void qemu_del_vlan_client(VLANClientState *vc)

{

    if (vc->vlan) {

        QTAILQ_REMOVE(&vc->vlan->clients, vc, next);

    } else {

        if (vc->send_queue) {

            qemu_del_net_queue(vc->send_queue);

        }

        QTAILQ_REMOVE(&non_vlan_clients, vc, next);

        if (vc->peer) {

            vc->peer->peer = NULL;

        }

    }



    if (vc->info->cleanup) {

        vc->info->cleanup(vc);

    }



    qemu_free(vc->name);

    qemu_free(vc->model);

    qemu_free(vc);

}
