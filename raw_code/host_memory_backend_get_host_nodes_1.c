host_memory_backend_get_host_nodes(Object *obj, Visitor *v, const char *name,

                                   void *opaque, Error **errp)

{

    HostMemoryBackend *backend = MEMORY_BACKEND(obj);

    uint16List *host_nodes = NULL;

    uint16List **node = &host_nodes;

    unsigned long value;



    value = find_first_bit(backend->host_nodes, MAX_NODES);



    node = host_memory_append_node(node, value);



    if (value == MAX_NODES) {

        goto out;

    }



    do {

        value = find_next_bit(backend->host_nodes, MAX_NODES, value + 1);

        if (value == MAX_NODES) {

            break;

        }



        node = host_memory_append_node(node, value);

    } while (true);



out:

    visit_type_uint16List(v, name, &host_nodes, errp);

}
