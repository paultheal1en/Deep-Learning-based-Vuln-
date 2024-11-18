static void kvm_ioapic_class_init(ObjectClass *klass, void *data)

{

    IOAPICCommonClass *k = IOAPIC_COMMON_CLASS(klass);

    DeviceClass *dc = DEVICE_CLASS(klass);



    k->realize   = kvm_ioapic_realize;

    k->pre_save  = kvm_ioapic_get;

    k->post_load = kvm_ioapic_put;

    dc->reset    = kvm_ioapic_reset;

    dc->props    = kvm_ioapic_properties;






}