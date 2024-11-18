int cpu_x86_register(X86CPU *cpu, const char *cpu_model)

{

    CPUX86State *env = &cpu->env;

    x86_def_t def1, *def = &def1;

    Error *error = NULL;

    char *name, *features;

    gchar **model_pieces;



    memset(def, 0, sizeof(*def));



    model_pieces = g_strsplit(cpu_model, ",", 2);

    if (!model_pieces[0]) {

        error_setg(&error, "Invalid/empty CPU model name");

        goto out;

    }

    name = model_pieces[0];

    features = model_pieces[1];



    if (cpu_x86_find_by_name(def, name) < 0) {

        error_setg(&error, "Unable to find CPU definition: %s", name);

        goto out;

    }



    if (kvm_enabled()) {

        def->kvm_features |= kvm_default_features;

    }

    def->ext_features |= CPUID_EXT_HYPERVISOR;



    object_property_set_str(OBJECT(cpu), def->vendor, "vendor", &error);

    object_property_set_int(OBJECT(cpu), def->level, "level", &error);

    object_property_set_int(OBJECT(cpu), def->family, "family", &error);

    object_property_set_int(OBJECT(cpu), def->model, "model", &error);

    object_property_set_int(OBJECT(cpu), def->stepping, "stepping", &error);

    env->cpuid_features = def->features;

    env->cpuid_ext_features = def->ext_features;

    env->cpuid_ext2_features = def->ext2_features;

    env->cpuid_ext3_features = def->ext3_features;

    object_property_set_int(OBJECT(cpu), def->xlevel, "xlevel", &error);

    env->cpuid_kvm_features = def->kvm_features;

    env->cpuid_svm_features = def->svm_features;

    env->cpuid_ext4_features = def->ext4_features;

    env->cpuid_7_0_ebx_features = def->cpuid_7_0_ebx_features;

    env->cpuid_xlevel2 = def->xlevel2;

    object_property_set_int(OBJECT(cpu), (int64_t)def->tsc_khz * 1000,

                            "tsc-frequency", &error);



    object_property_set_str(OBJECT(cpu), def->model_id, "model-id", &error);

    if (error) {

        goto out;

    }



    cpu_x86_parse_featurestr(cpu, features, &error);

out:

    g_strfreev(model_pieces);

    if (error) {

        fprintf(stderr, "%s\n", error_get_pretty(error));

        error_free(error);

        return -1;

    }

    return 0;

}
