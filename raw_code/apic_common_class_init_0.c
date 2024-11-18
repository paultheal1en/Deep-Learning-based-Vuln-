static void apic_common_class_init(ObjectClass *klass, void *data)

{

    ICCDeviceClass *idc = ICC_DEVICE_CLASS(klass);

    DeviceClass *dc = DEVICE_CLASS(klass);



    dc->vmsd = &vmstate_apic_common;

    dc->reset = apic_reset_common;

    dc->props = apic_properties_common;

    idc->realize = apic_common_realize;

    /*

     * Reason: APIC and CPU need to be wired up by

     * x86_cpu_apic_create()

     */

    dc->cannot_instantiate_with_device_add_yet = true;

}
