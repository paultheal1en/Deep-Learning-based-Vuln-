static void get_bit(Object *obj, Visitor *v, void *opaque,

                    const char *name, Error **errp)

{

    DeviceState *dev = DEVICE(obj);

    Property *prop = opaque;

    uint32_t *p = qdev_get_prop_ptr(dev, prop);

    bool value = (*p & qdev_get_prop_mask(prop)) != 0;



    visit_type_bool(v, &value, name, errp);

}
