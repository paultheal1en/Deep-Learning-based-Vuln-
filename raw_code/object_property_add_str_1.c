void object_property_add_str(Object *obj, const char *name,

                           char *(*get)(Object *, Error **),

                           void (*set)(Object *, const char *, Error **),

                           Error **errp)

{

    StringProperty *prop = g_malloc0(sizeof(*prop));



    prop->get = get;

    prop->set = set;



    object_property_add(obj, name, "string",

                        get ? property_get_str : NULL,

                        set ? property_set_str : NULL,

                        property_release_str,

                        prop, errp);

}
