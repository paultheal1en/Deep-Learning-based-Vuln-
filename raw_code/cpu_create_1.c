CPUState *cpu_create(const char *typename)

{

    Error *err = NULL;

    CPUState *cpu = CPU(object_new(typename));

    object_property_set_bool(OBJECT(cpu), true, "realized", &err);

    if (err != NULL) {

        error_report_err(err);

        object_unref(OBJECT(cpu));

        return NULL;

    }

    return cpu;

}
