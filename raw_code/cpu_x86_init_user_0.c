CPUX86State *cpu_x86_init_user(const char *cpu_model)

{

    Error *error = NULL;

    X86CPU *cpu;



    cpu = cpu_x86_create(cpu_model, NULL, &error);

    if (error) {

        goto out;

    }



    object_property_set_bool(OBJECT(cpu), true, "realized", &error);



out:

    if (error) {

        error_report("%s", error_get_pretty(error));

        error_free(error);

        if (cpu != NULL) {

            object_unref(OBJECT(cpu));

        }

        return NULL;

    }

    return &cpu->env;

}
