int kvm_irqchip_send_msi(KVMState *s, MSIMessage msg)

{

    struct kvm_msi msi;

    KVMMSIRoute *route;



    if (s->direct_msi) {

        msi.address_lo = (uint32_t)msg.address;

        msi.address_hi = msg.address >> 32;

        msi.data = le32_to_cpu(msg.data);

        msi.flags = 0;

        memset(msi.pad, 0, sizeof(msi.pad));



        return kvm_vm_ioctl(s, KVM_SIGNAL_MSI, &msi);

    }



    route = kvm_lookup_msi_route(s, msg);

    if (!route) {

        int virq;



        virq = kvm_irqchip_get_virq(s);

        if (virq < 0) {

            return virq;

        }



        route = g_malloc(sizeof(KVMMSIRoute));

        route->kroute.gsi = virq;

        route->kroute.type = KVM_IRQ_ROUTING_MSI;

        route->kroute.flags = 0;

        route->kroute.u.msi.address_lo = (uint32_t)msg.address;

        route->kroute.u.msi.address_hi = msg.address >> 32;

        route->kroute.u.msi.data = le32_to_cpu(msg.data);



        kvm_add_routing_entry(s, &route->kroute);

        kvm_irqchip_commit_routes(s);



        QTAILQ_INSERT_TAIL(&s->msi_hashtab[kvm_hash_msi(msg.data)], route,

                           entry);

    }



    assert(route->kroute.type == KVM_IRQ_ROUTING_MSI);



    return kvm_set_irq(s, route->kroute.gsi, 1);

}
