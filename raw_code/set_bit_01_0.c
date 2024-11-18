static void set_bit(Object *obj, Visitor *v, void *opaque,

                    const char *name, Error **errp)

{

    DeviceState *dev = DEVICE(obj);

    Property *prop = opaque;

    Error *local_err = NULL;

    bool value;



    if (dev->realized) {

        qdev_prop_set_after_realize(dev, name, errp);

        return;

    }



    visit_type_bool(v, &value, name, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        return;

    }

    bit_prop_set(dev, prop, value);

}
