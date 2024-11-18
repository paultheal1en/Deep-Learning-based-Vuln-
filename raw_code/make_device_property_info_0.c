static DevicePropertyInfo *make_device_property_info(ObjectClass *klass,

                                                     const char *name,

                                                     const char *default_type,

                                                     const char *description)

{

    DevicePropertyInfo *info;

    Property *prop;



    do {

        for (prop = DEVICE_CLASS(klass)->props; prop && prop->name; prop++) {

            if (strcmp(name, prop->name) != 0) {

                continue;

            }



            /*

             * TODO Properties without a parser are just for dirty hacks.

             * qdev_prop_ptr is the only such PropertyInfo.  It's marked

             * for removal.  This conditional should be removed along with

             * it.

             */

            if (!prop->info->set) {

                return NULL;           /* no way to set it, don't show */

            }



            info = g_malloc0(sizeof(*info));

            info->name = g_strdup(prop->name);

            info->type = g_strdup(prop->info->name);

            info->has_description = !!prop->info->description;

            info->description = g_strdup(prop->info->description);

            return info;

        }

        klass = object_class_get_parent(klass);

    } while (klass != object_class_by_name(TYPE_DEVICE));



    /* Not a qdev property, use the default type */

    info = g_malloc0(sizeof(*info));

    info->name = g_strdup(name);

    info->type = g_strdup(default_type);

    info->has_description = !!description;

    info->description = g_strdup(description);



    return info;

}
