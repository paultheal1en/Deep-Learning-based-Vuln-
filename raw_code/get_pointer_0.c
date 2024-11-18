static void get_pointer(Object *obj, Visitor *v, Property *prop,

                        const char *(*print)(void *ptr),

                        const char *name, Error **errp)

{

    DeviceState *dev = DEVICE(obj);

    void **ptr = qdev_get_prop_ptr(dev, prop);

    char *p;



    p = (char *) (*ptr ? print(*ptr) : "");

    visit_type_str(v, &p, name, errp);

}
