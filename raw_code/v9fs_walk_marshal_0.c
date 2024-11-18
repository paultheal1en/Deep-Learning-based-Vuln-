static int v9fs_walk_marshal(V9fsPDU *pdu, uint16_t nwnames, V9fsQID *qids)

{

    int i;

    size_t offset = 7;

    offset += pdu_marshal(pdu, offset, "w", nwnames);

    for (i = 0; i < nwnames; i++) {

        offset += pdu_marshal(pdu, offset, "Q", &qids[i]);

    }

    return offset;

}
