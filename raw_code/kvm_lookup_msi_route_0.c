static KVMMSIRoute *kvm_lookup_msi_route(KVMState *s, MSIMessage msg)

{

    unsigned int hash = kvm_hash_msi(msg.data);

    KVMMSIRoute *route;



    QTAILQ_FOREACH(route, &s->msi_hashtab[hash], entry) {

        if (route->kroute.u.msi.address_lo == (uint32_t)msg.address &&

            route->kroute.u.msi.address_hi == (msg.address >> 32) &&

            route->kroute.u.msi.data == msg.data) {

            return route;

        }

    }

    return NULL;

}
