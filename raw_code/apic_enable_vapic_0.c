void apic_enable_vapic(DeviceState *d, target_phys_addr_t paddr)

{

    APICCommonState *s = DO_UPCAST(APICCommonState, busdev.qdev, d);

    APICCommonClass *info = APIC_COMMON_GET_CLASS(s);



    s->vapic_paddr = paddr;

    info->vapic_base_update(s);

}
