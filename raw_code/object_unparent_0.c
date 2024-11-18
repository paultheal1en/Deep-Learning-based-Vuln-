void object_unparent(Object *obj)

{

    object_ref(obj);

    if (obj->parent) {

        object_property_del_child(obj->parent, obj, NULL);

    }

    if (obj->class->unparent) {

        (obj->class->unparent)(obj);

    }

    object_unref(obj);

}
