void qdev_init_nofail(DeviceState *dev)

{

    Error *err = NULL;



    assert(!dev->realized);




    object_property_set_bool(OBJECT(dev), true, "realized", &err);

    if (err) {

        error_reportf_err(err, "Initialization of device %s failed: ",

                          object_get_typename(OBJECT(dev)));

        exit(1);

    }


}