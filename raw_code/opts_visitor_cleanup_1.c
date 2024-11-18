opts_visitor_cleanup(OptsVisitor *ov)

{

    if (ov->unprocessed_opts != NULL) {

        g_hash_table_destroy(ov->unprocessed_opts);

    }

    g_free(ov->fake_id_opt);

    memset(ov, '\0', sizeof *ov);

}
