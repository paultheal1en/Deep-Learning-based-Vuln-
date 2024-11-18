static void kvm_arm_gicv3_realize(DeviceState *dev, Error **errp)

{

    GICv3State *s = KVM_ARM_GICV3(dev);

    KVMARMGICv3Class *kgc = KVM_ARM_GICV3_GET_CLASS(s);

    Error *local_err = NULL;




    DPRINTF("kvm_arm_gicv3_realize\n");



    kgc->parent_realize(dev, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        return;

    }



    if (s->security_extn) {

        error_setg(errp, "the in-kernel VGICv3 does not implement the "

                   "security extensions");

        return;

    }



    gicv3_init_irqs_and_mmio(s, kvm_arm_gicv3_set_irq, NULL);



    /* Try to create the device via the device control API */

    s->dev_fd = kvm_create_device(kvm_state, KVM_DEV_TYPE_ARM_VGIC_V3, false);

    if (s->dev_fd < 0) {

        error_setg_errno(errp, -s->dev_fd, "error creating in-kernel VGIC");

        return;

    }



    kvm_device_access(s->dev_fd, KVM_DEV_ARM_VGIC_GRP_NR_IRQS,

                      0, &s->num_irq, true);



    /* Tell the kernel to complete VGIC initialization now */

    kvm_device_access(s->dev_fd, KVM_DEV_ARM_VGIC_GRP_CTRL,

                      KVM_DEV_ARM_VGIC_CTRL_INIT, NULL, true);



    kvm_arm_register_device(&s->iomem_dist, -1, KVM_DEV_ARM_VGIC_GRP_ADDR,

                            KVM_VGIC_V3_ADDR_TYPE_DIST, s->dev_fd);

    kvm_arm_register_device(&s->iomem_redist, -1, KVM_DEV_ARM_VGIC_GRP_ADDR,

                            KVM_VGIC_V3_ADDR_TYPE_REDIST, s->dev_fd);



    /* Block migration of a KVM GICv3 device: the API for saving and restoring

     * the state in the kernel is not yet finalised in the kernel or

     * implemented in QEMU.

     */

    error_setg(&s->migration_blocker, "vGICv3 migration is not implemented");

    migrate_add_blocker(s->migration_blocker);



    if (kvm_has_gsi_routing()) {

        /* set up irq routing */

        kvm_init_irq_routing(kvm_state);

        for (i = 0; i < s->num_irq - GIC_INTERNAL; ++i) {

            kvm_irqchip_add_irq_route(kvm_state, i, 0, i);

        }



        kvm_gsi_routing_allowed = true;



        kvm_irqchip_commit_routes(kvm_state);

    }

}