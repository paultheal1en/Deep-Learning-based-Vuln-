static void set_pointer(Object *obj, Visitor *v, Property *prop,

                        int (*parse)(DeviceState *dev, const char *str, void **ptr),

                        const char *name, Error **errp)

{

    DeviceState *dev = DEVICE(obj);

    Error *local_err = NULL;

    void **ptr = qdev_get_prop_ptr(dev, prop);

    char *str;

    int ret;



    if (dev->state != DEV_STATE_CREATED) {

        error_set(errp, QERR_PERMISSION_DENIED);

        return;

    }



    visit_type_str(v, &str, name, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        return;

    }

    if (!*str) {

        g_free(str);

        *ptr = NULL;

        return;

    }

    ret = parse(dev, str, ptr);

    error_set_from_qdev_prop_error(errp, ret, dev, prop, str);

    g_free(str);

}
