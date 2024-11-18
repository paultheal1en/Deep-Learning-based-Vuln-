void net_client_uninit(NICInfo *nd)

{

    nd->vlan->nb_guest_devs--;

    nb_nics--;

    nd->used = 0;

    free((void *)nd->model);

}
