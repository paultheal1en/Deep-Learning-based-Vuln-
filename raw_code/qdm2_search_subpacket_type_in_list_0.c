static QDM2SubPNode *qdm2_search_subpacket_type_in_list(QDM2SubPNode *list,

                                                        int type)

{

    while (list != NULL && list->packet != NULL) {

        if (list->packet->type == type)

            return list;

        list = list->next;

    }

    return NULL;

}
