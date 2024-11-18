VLANState *qemu_find_vlan(int id, int allocate)

{

    VLANState **pvlan, *vlan;

    for(vlan = first_vlan; vlan != NULL; vlan = vlan->next) {

        if (vlan->id == id)

            return vlan;

    }

    if (!allocate) {

        return NULL;

    }

    vlan = qemu_mallocz(sizeof(VLANState));

    vlan->id = id;

    TAILQ_INIT(&vlan->send_queue);

    vlan->next = NULL;

    pvlan = &first_vlan;

    while (*pvlan != NULL)

        pvlan = &(*pvlan)->next;

    *pvlan = vlan;

    return vlan;

}
