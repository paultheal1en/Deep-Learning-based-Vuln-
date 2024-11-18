static void qdev_prop_set_globals_for_type(DeviceState *dev,

                                           const char *typename)

{

    GList *l;



    for (l = global_props; l; l = l->next) {

        GlobalProperty *prop = l->data;

        Error *err = NULL;



        if (strcmp(typename, prop->driver) != 0) {

            continue;

        }

        prop->used = true;

        object_property_parse(OBJECT(dev), prop->value, prop->property, &err);

        if (err != NULL) {

            error_prepend(&err, "can't apply global %s.%s=%s: ",

                          prop->driver, prop->property, prop->value);

            if (!dev->hotplugged && prop->errp) {

                error_propagate(prop->errp, err);

            } else {

                assert(prop->user_provided);

                warn_report_err(err);

            }

        }

    }

}
