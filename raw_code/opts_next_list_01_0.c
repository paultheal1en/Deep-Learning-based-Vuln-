opts_next_list(Visitor *v, GenericList **list, size_t size)

{

    OptsVisitor *ov = to_ov(v);

    GenericList **link;



    switch (ov->list_mode) {

    case LM_STARTED:

        ov->list_mode = LM_IN_PROGRESS;

        link = list;

        break;



    case LM_SIGNED_INTERVAL:

    case LM_UNSIGNED_INTERVAL:

        link = &(*list)->next;



        if (ov->list_mode == LM_SIGNED_INTERVAL) {

            if (ov->range_next.s < ov->range_limit.s) {

                ++ov->range_next.s;

                break;

            }

        } else if (ov->range_next.u < ov->range_limit.u) {

            ++ov->range_next.u;

            break;

        }

        ov->list_mode = LM_IN_PROGRESS;

        /* range has been completed, fall through in order to pop option */



    case LM_IN_PROGRESS: {

        const QemuOpt *opt;



        opt = g_queue_pop_head(ov->repeated_opts);

        if (g_queue_is_empty(ov->repeated_opts)) {

            g_hash_table_remove(ov->unprocessed_opts, opt->name);

            return NULL;

        }

        link = &(*list)->next;

        break;

    }



    default:

        abort();

    }



    *link = g_malloc0(size);

    return *link;

}
