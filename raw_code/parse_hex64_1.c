static int parse_hex64(DeviceState *dev, Property *prop, const char *str)

{

    uint64_t *ptr = qdev_get_prop_ptr(dev, prop);



    if (sscanf(str, "%" PRIx64, ptr) != 1)

        return -EINVAL;

    return 0;

}
