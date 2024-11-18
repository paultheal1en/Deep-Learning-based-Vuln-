start_list(Visitor *v, const char *name, GenericList **list, size_t size,

           Error **errp)

{

    StringInputVisitor *siv = to_siv(v);



    /* We don't support visits without a list */

    assert(list);



    if (parse_str(siv, name, errp) < 0) {

        *list = NULL;

        return;

    }



    siv->cur_range = g_list_first(siv->ranges);

    if (siv->cur_range) {

        Range *r = siv->cur_range->data;

        if (r) {

            siv->cur = r->begin;

        }

        *list = g_malloc0(size);

    } else {

        *list = NULL;

    }

}
