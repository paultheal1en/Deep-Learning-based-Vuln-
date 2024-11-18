static void process_synthesis_subpackets(QDM2Context *q, QDM2SubPNode *list)

{

    QDM2SubPNode *nodes[4];



    nodes[0] = qdm2_search_subpacket_type_in_list(list, 9);

    if (nodes[0] != NULL)

        process_subpacket_9(q, nodes[0]);



    nodes[1] = qdm2_search_subpacket_type_in_list(list, 10);

    if (nodes[1] != NULL)

        process_subpacket_10(q, nodes[1]);

    else

        process_subpacket_10(q, NULL);



    nodes[2] = qdm2_search_subpacket_type_in_list(list, 11);

    if (nodes[0] != NULL && nodes[1] != NULL && nodes[2] != NULL)

        process_subpacket_11(q, nodes[2]);

    else

        process_subpacket_11(q, NULL);



    nodes[3] = qdm2_search_subpacket_type_in_list(list, 12);

    if (nodes[0] != NULL && nodes[1] != NULL && nodes[3] != NULL)

        process_subpacket_12(q, nodes[3]);

    else

        process_subpacket_12(q, NULL);

}
