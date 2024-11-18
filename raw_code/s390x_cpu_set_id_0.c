static void s390x_cpu_set_id(Object *obj, Visitor *v, const char *name,

                             void *opaque, Error **errp)

{

    S390CPU *cpu = S390_CPU(obj);

    DeviceState *dev = DEVICE(obj);

    const int64_t min = 0;

    const int64_t max = UINT32_MAX;

    Error *err = NULL;

    int64_t value;



    if (dev->realized) {

        error_setg(errp, "Attempt to set property '%s' on '%s' after "

                   "it was realized", name, object_get_typename(obj));

        return;

    }



    visit_type_int(v, name, &value, &err);

    if (err) {

        error_propagate(errp, err);

        return;

    }

    if (value < min || value > max) {

        error_setg(errp, "Property %s.%s doesn't take value %" PRId64

                   " (minimum: %" PRId64 ", maximum: %" PRId64 ")" ,

                   object_get_typename(obj), name, value, min, max);

        return;

    }

    cpu->id = value;

}
