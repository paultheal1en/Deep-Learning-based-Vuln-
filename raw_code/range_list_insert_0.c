GList *range_list_insert(GList *list, Range *data)

{

    GList *l;



    /* Range lists require no empty ranges */

    assert(data->begin < data->end || (data->begin && !data->end));



    /* Skip all list elements strictly less than data */

    for (l = list; l && range_compare(l->data, data) < 0; l = l->next) {

    }



    if (!l || range_compare(l->data, data) > 0) {

        /* Rest of the list (if any) is strictly greater than @data */

        return g_list_insert_before(list, l, data);

    }



    /* Current list element overlaps @data, merge the two */

    range_extend(l->data, data);

    g_free(data);



    /* Merge any subsequent list elements that now also overlap */

    while (l->next && range_compare(l->data, l->next->data) == 0) {

        GList *new_l;



        range_extend(l->data, l->next->data);

        g_free(l->next->data);

        new_l = g_list_delete_link(list, l->next);

        assert(new_l == list);

    }



    return list;

}
