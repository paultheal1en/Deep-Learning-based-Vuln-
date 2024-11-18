static ObjectClass *alpha_cpu_class_by_name(const char *cpu_model)

{

    ObjectClass *oc = NULL;

    char *typename;

    int i;



    if (cpu_model == NULL) {

        return NULL;

    }



    oc = object_class_by_name(cpu_model);

    if (oc != NULL && object_class_dynamic_cast(oc, TYPE_ALPHA_CPU) != NULL &&

        !object_class_is_abstract(oc)) {

        return oc;

    }



    for (i = 0; i < ARRAY_SIZE(alpha_cpu_aliases); i++) {

        if (strcmp(cpu_model, alpha_cpu_aliases[i].alias) == 0) {

            oc = object_class_by_name(alpha_cpu_aliases[i].typename);

            assert(oc != NULL && !object_class_is_abstract(oc));

            return oc;

        }

    }



    typename = g_strdup_printf("%s-" TYPE_ALPHA_CPU, cpu_model);

    oc = object_class_by_name(typename);

    g_free(typename);

    if (oc != NULL && object_class_is_abstract(oc)) {

        oc = NULL;

    }

    return oc;

}
