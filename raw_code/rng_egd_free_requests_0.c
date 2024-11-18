static void rng_egd_free_requests(RngEgd *s)

{

    GSList *i;



    for (i = s->parent.requests; i; i = i->next) {

        rng_egd_free_request(i->data);

    }



    g_slist_free(s->parent.requests);

    s->parent.requests = NULL;

}
