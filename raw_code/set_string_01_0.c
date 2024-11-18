static void set_string(Object *obj, Visitor *v, void *opaque,

                       const char *name, Error **errp)

{

    DeviceState *dev = DEVICE(obj);

    Property *prop = opaque;

    char **ptr = qdev_get_prop_ptr(dev, prop);

    Error *local_err = NULL;

    char *str;



    if (dev->realized) {

        qdev_prop_set_after_realize(dev, name, errp);

        return;

    }



    visit_type_str(v, &str, name, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        return;

    }

    if (*ptr) {

        g_free(*ptr);

    }

    *ptr = str;

}
