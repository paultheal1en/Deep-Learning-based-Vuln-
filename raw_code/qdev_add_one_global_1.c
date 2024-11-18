static int qdev_add_one_global(QemuOpts *opts, void *opaque)

{

    GlobalProperty *g;

    ObjectClass *oc;



    g = g_malloc0(sizeof(*g));

    g->driver   = qemu_opt_get(opts, "driver");

    g->property = qemu_opt_get(opts, "property");

    g->value    = qemu_opt_get(opts, "value");

    oc = object_class_by_name(g->driver);

    if (oc) {

        DeviceClass *dc = DEVICE_CLASS(oc);



        if (dc->hotpluggable) {

            /* If hotpluggable then skip not_used checking. */

            g->not_used = false;

        } else {

            /* Maybe a typo. */

            g->not_used = true;

        }

    } else {

        /* Maybe a typo. */

        g->not_used = true;

    }

    qdev_prop_register_global(g);

    return 0;

}
