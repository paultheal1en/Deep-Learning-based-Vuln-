void object_delete(Object *obj)

{

    object_unparent(obj);

    g_assert(obj->ref == 1);

    object_unref(obj);

    g_free(obj);

}
