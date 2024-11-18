void object_property_add(Object *obj, const char *name, const char *type,

                         ObjectPropertyAccessor *get,

                         ObjectPropertyAccessor *set,

                         ObjectPropertyRelease *release,

                         void *opaque, Error **errp)

{

    ObjectProperty *prop;



    QTAILQ_FOREACH(prop, &obj->properties, node) {

        if (strcmp(prop->name, name) == 0) {

            error_setg(errp, "attempt to add duplicate property '%s'"

                       " to object (type '%s')", name,

                       object_get_typename(obj));

            return;

        }

    }



    prop = g_malloc0(sizeof(*prop));



    prop->name = g_strdup(name);

    prop->type = g_strdup(type);



    prop->get = get;

    prop->set = set;

    prop->release = release;

    prop->opaque = opaque;



    QTAILQ_INSERT_TAIL(&obj->properties, prop, node);

}
