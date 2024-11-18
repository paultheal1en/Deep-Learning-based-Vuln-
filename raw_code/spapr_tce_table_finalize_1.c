static void spapr_tce_table_finalize(Object *obj)

{

    sPAPRTCETable *tcet = SPAPR_TCE_TABLE(obj);



    QLIST_REMOVE(tcet, list);



    if (!kvm_enabled() ||

        (kvmppc_remove_spapr_tce(tcet->table, tcet->fd,

                                 tcet->nb_table) != 0)) {

        g_free(tcet->table);

    }

}
