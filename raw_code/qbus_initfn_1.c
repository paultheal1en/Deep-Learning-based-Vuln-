static void qbus_initfn(Object *obj)

{

    BusState *bus = BUS(obj);



    QTAILQ_INIT(&bus->children);

    object_property_add_link(obj, QDEV_HOTPLUG_HANDLER_PROPERTY,

                             TYPE_HOTPLUG_HANDLER,

                             (Object **)&bus->hotplug_handler, NULL);

    object_property_add_bool(obj, "realized",

                             bus_get_realized, bus_set_realized, NULL);

}
