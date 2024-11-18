S390CPU *cpu_s390x_create(const char *cpu_model, Error **errp)

{

    static bool features_parsed;

    char *name, *features;

    const char *typename;

    ObjectClass *oc;

    CPUClass *cc;



    name = g_strdup(cpu_model);

    features = strchr(name, ',');

    if (features) {

        features[0] = 0;

        features++;

    }



    oc = cpu_class_by_name(TYPE_S390_CPU, name);

    if (!oc) {

        error_setg(errp, "Unknown CPU definition \'%s\'", name);

        g_free(name);

        return NULL;

    }

    typename = object_class_get_name(oc);



    if (!features_parsed) {

        features_parsed = true;

        cc = CPU_CLASS(oc);

        cc->parse_features(typename, features, errp);

    }

    g_free(name);



    if (*errp) {

        return NULL;

    }

    return S390_CPU(CPU(object_new(typename)));

}
