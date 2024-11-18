static void arm_gic_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    SysBusDeviceClass *sbc = SYS_BUS_DEVICE_CLASS(klass);

    ARMGICClass *agc = ARM_GIC_CLASS(klass);

    agc->parent_init = sbc->init;

    sbc->init = arm_gic_init;

    dc->no_user = 1;

}
