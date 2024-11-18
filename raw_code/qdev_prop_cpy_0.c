static void qdev_prop_cpy(DeviceState *dev, Property *props, void *src)

{

    if (props->info->type == PROP_TYPE_BIT) {

        bool *defval = src;

        bit_prop_set(dev, props, *defval);

    } else {

        char *dst = qdev_get_prop_ptr(dev, props);

        memcpy(dst, src, props->info->size);

    }

}
