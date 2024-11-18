ObjectClass *object_class_dynamic_cast(ObjectClass *class,

                                       const char *typename)

{

    TypeImpl *target_type = type_get_by_name(typename);

    TypeImpl *type = class->type;

    ObjectClass *ret = NULL;



    if (type->num_interfaces && type_is_ancestor(target_type, type_interface)) {

        int found = 0;

        GSList *i;



        for (i = class->interfaces; i; i = i->next) {

            ObjectClass *target_class = i->data;



            if (type_is_ancestor(target_class->type, target_type)) {

                ret = target_class;

                found++;

            }

         }



        /* The match was ambiguous, don't allow a cast */

        if (found > 1) {

            ret = NULL;

        }

    } else if (type_is_ancestor(type, target_type)) {

        ret = class;

    }



    return ret;

}
