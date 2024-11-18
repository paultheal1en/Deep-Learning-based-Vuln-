static int object_create(QemuOpts *opts, void *opaque)

{

    const char *type = qemu_opt_get(opts, "qom-type");

    const char *id = qemu_opts_id(opts);

    Object *obj;



    g_assert(type != NULL);



    if (id == NULL) {

        qerror_report(QERR_MISSING_PARAMETER, "id");

        return -1;

    }



    obj = object_new(type);

    if (qemu_opt_foreach(opts, object_set_property, obj, 1) < 0) {

        return -1;

    }



    object_property_add_child(container_get(object_get_root(), "/objects"),

                              id, obj, NULL);



    return 0;

}
