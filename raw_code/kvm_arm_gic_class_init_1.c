static void kvm_arm_gic_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    ARMGICCommonClass *agcc = ARM_GIC_COMMON_CLASS(klass);

    KVMARMGICClass *kgc = KVM_ARM_GIC_CLASS(klass);



    agcc->pre_save = kvm_arm_gic_get;

    agcc->post_load = kvm_arm_gic_put;

    kgc->parent_realize = dc->realize;

    kgc->parent_reset = dc->reset;

    dc->realize = kvm_arm_gic_realize;

    dc->reset = kvm_arm_gic_reset;

    dc->no_user = 1;

}
