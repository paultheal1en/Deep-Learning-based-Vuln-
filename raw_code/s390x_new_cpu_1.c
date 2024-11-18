S390CPU *s390x_new_cpu(const char *typename, uint32_t core_id, Error **errp)

{

    S390CPU *cpu = S390_CPU(object_new(typename));

    Error *err = NULL;



    object_property_set_int(OBJECT(cpu), core_id, "core-id", &err);

    if (err != NULL) {

        goto out;

    }

    object_property_set_bool(OBJECT(cpu), true, "realized", &err);



out:

    if (err) {

        error_propagate(errp, err);

        object_unref(OBJECT(cpu));

        cpu = NULL;

    }

    return cpu;

}
