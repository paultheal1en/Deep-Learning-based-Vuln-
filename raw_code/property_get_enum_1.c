static void property_get_enum(Object *obj, Visitor *v, void *opaque,

                              const char *name, Error **errp)

{

    EnumProperty *prop = opaque;

    int value;



    value = prop->get(obj, errp);

    visit_type_enum(v, &value, prop->strings, NULL, name, errp);

}
