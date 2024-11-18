static X86CPU *pc_new_cpu(const char *cpu_model, int64_t apic_id,

                          DeviceState *icc_bridge, Error **errp)

{

    X86CPU *cpu = NULL;

    Error *local_err = NULL;



    if (icc_bridge == NULL) {

        error_setg(&local_err, "Invalid icc-bridge value");

        goto out;

    }



    cpu = cpu_x86_create(cpu_model, &local_err);

    if (local_err != NULL) {

        goto out;

    }



    qdev_set_parent_bus(DEVICE(cpu), qdev_get_child_bus(icc_bridge, "icc"));



    object_property_set_int(OBJECT(cpu), apic_id, "apic-id", &local_err);

    object_property_set_bool(OBJECT(cpu), true, "realized", &local_err);



out:

    if (local_err) {

        error_propagate(errp, local_err);

        object_unref(OBJECT(cpu));

        cpu = NULL;

    }

    return cpu;

}
