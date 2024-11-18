GList *range_list_insert(GList *list, Range *data)

{

    GList *l, *next = NULL;

    Range *r, *nextr;



    if (!list) {

        list = g_list_insert_sorted(list, data, range_compare);

        return list;

    }



    nextr = data;

    l = list;

    while (l && l != next && nextr) {

        r = l->data;

        if (ranges_can_merge(r, nextr)) {

            range_merge(r, nextr);

            l = g_list_remove_link(l, next);

            next = g_list_next(l);

            if (next) {

                nextr = next->data;

            } else {

                nextr = NULL;

            }

        } else {

            l = g_list_next(l);

        }

    }



    if (!l) {

        list = g_list_insert_sorted(list, data, range_compare);

    }



    return list;

}
