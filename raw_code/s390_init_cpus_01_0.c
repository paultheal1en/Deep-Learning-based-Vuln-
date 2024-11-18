void s390_init_cpus(MachineState *machine)

{

    int i;

    gchar *name;



    if (machine->cpu_model == NULL) {

        machine->cpu_model = "host";

    }



    cpu_states = g_malloc0(sizeof(S390CPU *) * max_cpus);



    for (i = 0; i < max_cpus; i++) {

        name = g_strdup_printf("cpu[%i]", i);

        object_property_add_link(OBJECT(machine), name, TYPE_S390_CPU,

                                 (Object **) &cpu_states[i],

                                 object_property_allow_set_link,

                                 OBJ_PROP_LINK_UNREF_ON_RELEASE,

                                 &error_abort);

        g_free(name);

    }



    for (i = 0; i < smp_cpus; i++) {

        cpu_s390x_init(machine->cpu_model);

    }

}
