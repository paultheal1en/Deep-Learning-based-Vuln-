uint64_t cpu_get_apic_base(DeviceState *d)

{

    if (d) {

        APICCommonState *s = APIC_COMMON(d);

        trace_cpu_get_apic_base((uint64_t)s->apicbase);

        return s->apicbase;

    } else {

        trace_cpu_get_apic_base(0);

        return 0;

    }

}
