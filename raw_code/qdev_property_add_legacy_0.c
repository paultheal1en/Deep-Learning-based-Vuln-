void qdev_property_add_legacy(DeviceState *dev, Property *prop,

                              Error **errp)

{

    gchar *type;



    type = g_strdup_printf("legacy<%s>", prop->info->name);



    qdev_property_add(dev, prop->name, type,

                      qdev_get_legacy_property,

                      qdev_set_legacy_property,

                      NULL,

                      prop, errp);



    g_free(type);

}
