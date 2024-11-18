void qdev_prop_allow_set_link_before_realize(Object *obj, const char *name,

                                             Object *val, Error **errp)

{

    DeviceState *dev = DEVICE(obj);



    if (dev->realized) {

        error_setg(errp, "Attempt to set link property '%s' on device '%s' "

                   "(type '%s') after it was realized",

                   name, dev->id, object_get_typename(obj));

    }

}
