void qdev_prop_set_globals(DeviceState *dev)

{

    ObjectClass *class = object_get_class(OBJECT(dev));



    do {

        qdev_prop_set_globals_for_type(dev, object_class_get_name(class));

        class = object_class_get_parent(class);

    } while (class);

}
