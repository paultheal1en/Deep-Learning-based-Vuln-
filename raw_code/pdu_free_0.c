void pdu_free(V9fsPDU *pdu)

{

    if (pdu) {

        V9fsState *s = pdu->s;

        /*

         * Cancelled pdu are added back to the freelist

         * by flush request .

         */

        if (!pdu->cancelled) {

            QLIST_REMOVE(pdu, next);

            QLIST_INSERT_HEAD(&s->free_list, pdu, next);

        }

    }

}
