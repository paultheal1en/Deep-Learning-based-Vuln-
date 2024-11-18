static int parse_numa(void *opaque, QemuOpts *opts, Error **errp)

{

    NumaOptions *object = NULL;

    Error *err = NULL;



    {

        Visitor *v = opts_visitor_new(opts);

        visit_type_NumaOptions(v, NULL, &object, &err);

        visit_free(v);

    }



    if (err) {

        goto error;

    }



    switch (object->type) {

    case NUMA_OPTIONS_KIND_NODE:

        numa_node_parse(object->u.node.data, opts, &err);

        if (err) {

            goto error;

        }

        nb_numa_nodes++;

        break;

    default:

        abort();

    }



    return 0;



error:

    error_report_err(err);

    qapi_free_NumaOptions(object);



    return -1;

}
