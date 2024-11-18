void qdev_prop_set_defaults(DeviceState *dev, Property *props)

{

    if (!props)

        return;

    while (props->name) {

        if (props->defval) {

            qdev_prop_cpy(dev, props, props->defval);

        }

        props++;

    }

}
