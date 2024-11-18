static bool object_is_type(Object *obj, const char *typename)

{

    TypeImpl *target_type = type_get_by_name(typename);

    TypeImpl *type = obj->class->type;

    GSList *i;



    /* Check if typename is a direct ancestor of type */

    while (type) {

        if (type == target_type) {

            return true;

        }



        type = type_get_parent(type);

    }



    /* Check if obj has an interface of typename */

    for (i = obj->interfaces; i; i = i->next) {

        Interface *iface = i->data;



        if (object_is_type(OBJECT(iface), typename)) {

            return true;

        }

    }



    return false;

}
