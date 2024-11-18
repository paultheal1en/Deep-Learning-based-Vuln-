static void kvm_apic_realize(DeviceState *dev, Error **errp)

{

    APICCommonState *s = APIC_COMMON(dev);



    memory_region_init_io(&s->io_memory, NULL, &kvm_apic_io_ops, s, "kvm-apic-msi",

                          APIC_SPACE_SIZE);



    if (kvm_has_gsi_routing()) {

        msi_nonbroken = true;

    }

}
