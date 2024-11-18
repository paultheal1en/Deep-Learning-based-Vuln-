static int query_memdev(Object *obj, void *opaque)

{

    MemdevList **list = opaque;

    Error *err = NULL;



    if (object_dynamic_cast(obj, TYPE_MEMORY_BACKEND)) {

        MemdevList *m = g_malloc0(sizeof(*m));



        m->value = g_malloc0(sizeof(*m->value));



        m->value->size = object_property_get_int(obj, "size",

                                                 &err);

        if (err) {

            goto error;

        }



        m->value->merge = object_property_get_bool(obj, "merge",

                                                   &err);

        if (err) {

            goto error;

        }



        m->value->dump = object_property_get_bool(obj, "dump",

                                                  &err);

        if (err) {

            goto error;

        }



        m->value->prealloc = object_property_get_bool(obj,

                                                      "prealloc", &err);

        if (err) {

            goto error;

        }



        m->value->policy = object_property_get_enum(obj,

                                                    "policy",

                                                    HostMemPolicy_lookup,

                                                    &err);

        if (err) {

            goto error;

        }



        object_property_get_uint16List(obj, "host-nodes",

                                       &m->value->host_nodes, &err);

        if (err) {

            goto error;

        }



        m->next = *list;

        *list = m;

    }



    return 0;

error:

    return -1;

}
