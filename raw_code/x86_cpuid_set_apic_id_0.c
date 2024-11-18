static void x86_cpuid_set_apic_id(Object *obj, Visitor *v, const char *name,

                                  void *opaque, Error **errp)

{

    X86CPU *cpu = X86_CPU(obj);

    DeviceState *dev = DEVICE(obj);

    const int64_t min = 0;

    const int64_t max = UINT32_MAX;

    Error *error = NULL;

    int64_t value;



    if (dev->realized) {

        error_setg(errp, "Attempt to set property '%s' on '%s' after "

                   "it was realized", name, object_get_typename(obj));

        return;

    }



    visit_type_int(v, name, &value, &error);

    if (error) {

        error_propagate(errp, error);

        return;

    }

    if (value < min || value > max) {

        error_setg(errp, "Property %s.%s doesn't take value %" PRId64

                   " (minimum: %" PRId64 ", maximum: %" PRId64 ")" ,

                   object_get_typename(obj), name, value, min, max);

        return;

    }



    if ((value != cpu->apic_id) && cpu_exists(value)) {

        error_setg(errp, "CPU with APIC ID %" PRIi64 " exists", value);

        return;

    }

    cpu->apic_id = value;

}
