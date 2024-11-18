static void init_event_facility(Object *obj)

{

    SCLPEventFacility *event_facility = EVENT_FACILITY(obj);

    DeviceState *sdev = DEVICE(obj);



    /* Spawn a new bus for SCLP events */

    qbus_create_inplace(&event_facility->sbus, sizeof(event_facility->sbus),

                        TYPE_SCLP_EVENTS_BUS, sdev, NULL);



    object_initialize(&event_facility->quiesce_event, sizeof(SCLPEvent),

                      TYPE_SCLP_QUIESCE);

    qdev_set_parent_bus(DEVICE(&event_facility->quiesce_event),

                        &event_facility->sbus.qbus);

    object_initialize(&event_facility->cpu_hotplug_event, sizeof(SCLPEvent),

                      TYPE_SCLP_CPU_HOTPLUG);

    qdev_set_parent_bus(DEVICE(&event_facility->cpu_hotplug_event),

                        &event_facility->sbus.qbus);

    /* the facility will automatically realize the devices via the bus */

}
