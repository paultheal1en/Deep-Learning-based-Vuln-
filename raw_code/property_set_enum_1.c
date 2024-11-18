static void property_set_enum(Object *obj, Visitor *v, void *opaque,

                              const char *name, Error **errp)

{

    EnumProperty *prop = opaque;

    int value;



    visit_type_enum(v, &value, prop->strings, NULL, name, errp);

    prop->set(obj, value, errp);

}
