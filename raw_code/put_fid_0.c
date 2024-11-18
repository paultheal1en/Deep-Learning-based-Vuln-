static void put_fid(V9fsPDU *pdu, V9fsFidState *fidp)

{

    BUG_ON(!fidp->ref);

    fidp->ref--;

    /*

     * Don't free the fid if it is in reclaim list

     */

    if (!fidp->ref && fidp->clunked) {

        if (fidp->fid == pdu->s->root_fid) {

            /*

             * if the clunked fid is root fid then we

             * have unmounted the fs on the client side.

             * delete the migration blocker. Ideally, this

             * should be hooked to transport close notification

             */

            if (pdu->s->migration_blocker) {

                migrate_del_blocker(pdu->s->migration_blocker);

                error_free(pdu->s->migration_blocker);

                pdu->s->migration_blocker = NULL;

            }

        }

        free_fid(pdu, fidp);

    }

}
