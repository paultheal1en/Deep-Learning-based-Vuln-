static GenericList *next_list(Visitor *v, GenericList *tail, size_t size)

{

    StringInputVisitor *siv = to_siv(v);

    Range *r;



    if (!siv->ranges || !siv->cur_range) {

        return NULL;

    }



    r = siv->cur_range->data;

    if (!r) {

        return NULL;

    }



    if (siv->cur < r->begin || siv->cur >= r->end) {

        siv->cur_range = g_list_next(siv->cur_range);

        if (!siv->cur_range) {

            return NULL;

        }

        r = siv->cur_range->data;

        if (!r) {

            return NULL;

        }

        siv->cur = r->begin;

    }



    tail->next = g_malloc0(size);

    return tail->next;

}
