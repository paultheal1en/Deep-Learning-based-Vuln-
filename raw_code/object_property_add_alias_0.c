void object_property_add_alias(Object *obj, const char *name,

                               Object *target_obj, const char *target_name,

                               Error **errp)

{

    AliasProperty *prop;

    ObjectProperty *target_prop;



    target_prop = object_property_find(target_obj, target_name, errp);

    if (!target_prop) {

        return;

    }



    prop = g_malloc(sizeof(*prop));

    prop->target_obj = target_obj;

    prop->target_name = target_name;



    object_property_add(obj, name, target_prop->type,

                        property_get_alias,

                        property_set_alias,

                        property_release_alias,

                        prop, errp);

}
