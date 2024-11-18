static void powerpc_get_compat(Object *obj, Visitor *v, const char *name,

                               void *opaque, Error **errp)

{

    char *value = (char *)"";

    Property *prop = opaque;

    uint32_t *max_compat = qdev_get_prop_ptr(DEVICE(obj), prop);



    switch (*max_compat) {

    case CPU_POWERPC_LOGICAL_2_05:

        value = (char *)"power6";

        break;

    case CPU_POWERPC_LOGICAL_2_06:

        value = (char *)"power7";

        break;

    case CPU_POWERPC_LOGICAL_2_07:

        value = (char *)"power8";

        break;

    case 0:

        break;

    default:

        error_report("Internal error: compat is set to %x", *max_compat);

        abort();

        break;

    }



    visit_type_str(v, name, &value, errp);

}
