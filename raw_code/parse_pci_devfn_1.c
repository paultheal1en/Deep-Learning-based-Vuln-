static int parse_pci_devfn(DeviceState *dev, Property *prop, const char *str)

{

    uint32_t *ptr = qdev_get_prop_ptr(dev, prop);

    unsigned int slot, fn, n;



    if (sscanf(str, "%x.%x%n", &slot, &fn, &n) != 2) {

        fn = 0;

        if (sscanf(str, "%x%n", &slot, &n) != 1) {

            return -EINVAL;

        }

    }

    if (str[n] != '\0')

        return -EINVAL;

    if (fn > 7)

        return -EINVAL;

    if (slot > 31)

        return -EINVAL;

    *ptr = slot << 3 | fn;

    return 0;

}
