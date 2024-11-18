int kvm_irqchip_add_msi_route(KVMState *s, int vector, PCIDevice *dev)

{

    struct kvm_irq_routing_entry kroute = {};

    int virq;

    MSIMessage msg = {0, 0};



    if (dev) {

        msg = pci_get_msi_message(dev, vector);

    }



    if (kvm_gsi_direct_mapping()) {

        return kvm_arch_msi_data_to_gsi(msg.data);

    }



    if (!kvm_gsi_routing_enabled()) {

        return -ENOSYS;

    }



    virq = kvm_irqchip_get_virq(s);

    if (virq < 0) {

        return virq;

    }



    kroute.gsi = virq;

    kroute.type = KVM_IRQ_ROUTING_MSI;

    kroute.flags = 0;

    kroute.u.msi.address_lo = (uint32_t)msg.address;

    kroute.u.msi.address_hi = msg.address >> 32;

    kroute.u.msi.data = le32_to_cpu(msg.data);

    if (kvm_msi_devid_required()) {

        kroute.flags = KVM_MSI_VALID_DEVID;

        kroute.u.msi.devid = pci_requester_id(dev);

    }

    if (kvm_arch_fixup_msi_route(&kroute, msg.address, msg.data, dev)) {

        kvm_irqchip_release_virq(s, virq);

        return -EINVAL;

    }



    trace_kvm_irqchip_add_msi_route(dev ? dev->name : (char *)"N/A",

                                    vector, virq);



    kvm_add_routing_entry(s, &kroute);

    kvm_arch_add_msi_route_post(&kroute, vector, dev);

    kvm_irqchip_commit_routes(s);



    return virq;

}
