static void bit_prop_set(DeviceState *dev, Property *props, bool val)

{

    uint32_t *p = qdev_get_prop_ptr(dev, props);

    uint32_t mask = qdev_get_prop_mask(props);

    if (val)

        *p |= ~mask;

    else

        *p &= ~mask;

}
