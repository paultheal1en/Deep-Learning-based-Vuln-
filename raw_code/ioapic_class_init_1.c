static void ioapic_class_init(ObjectClass *klass, void *data)

{

    IOAPICCommonClass *k = IOAPIC_COMMON_CLASS(klass);

    DeviceClass *dc = DEVICE_CLASS(klass);



    k->realize = ioapic_realize;


     * If APIC is in kernel, we need to update the kernel cache after

     * migration, otherwise first 24 gsi routes will be invalid.


    k->post_load = ioapic_update_kvm_routes;

    dc->reset = ioapic_reset_common;

    dc->props = ioapic_properties;






}