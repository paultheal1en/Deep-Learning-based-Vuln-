static void qdev_print_devinfo(ObjectClass *klass, void *opaque)

{

    DeviceClass *dc;

    bool *show_no_user = opaque;



    dc = (DeviceClass *)object_class_dynamic_cast(klass, TYPE_DEVICE);



    if (!dc || (show_no_user && !*show_no_user && dc->no_user)) {

        return;

    }



    error_printf("name \"%s\"", object_class_get_name(klass));

    if (dc->bus_info) {

        error_printf(", bus %s", dc->bus_info->name);

    }

    if (dc->alias) {

        error_printf(", alias \"%s\"", dc->alias);

    }

    if (dc->desc) {

        error_printf(", desc \"%s\"", dc->desc);

    }

    if (dc->no_user) {

        error_printf(", no-user");

    }

    error_printf("\n");

}
