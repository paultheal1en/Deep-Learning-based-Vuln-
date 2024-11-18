static int apic_init_common(SysBusDevice *dev)

{

    APICCommonState *s = APIC_COMMON(dev);

    APICCommonClass *info;

    static DeviceState *vapic;

    static int apic_no;



    if (apic_no >= MAX_APICS) {

        return -1;

    }

    s->idx = apic_no++;



    info = APIC_COMMON_GET_CLASS(s);

    info->init(s);



    sysbus_init_mmio(dev, &s->io_memory);



    if (!vapic && s->vapic_control & VAPIC_ENABLE_MASK) {

        vapic = sysbus_create_simple("kvmvapic", -1, NULL);

    }

    s->vapic = vapic;

    if (apic_report_tpr_access && info->enable_tpr_reporting) {

        info->enable_tpr_reporting(s, true);

    }



    return 0;

}
