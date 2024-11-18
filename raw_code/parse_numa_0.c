static int parse_numa(void *opaque, QemuOpts *opts, Error **errp)

{

    NumaOptions *object = NULL;

    Error *err = NULL;



    {

        OptsVisitor *ov = opts_visitor_new(opts);

        visit_type_NumaOptions(opts_get_visitor(ov), NULL, &object, &err);

        opts_visitor_cleanup(ov);

    }



    if (err) {

        goto error;

    }



    switch (object->type) {

    case NUMA_OPTIONS_KIND_NODE:

        numa_node_parse(object->u.node, opts, &err);

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
