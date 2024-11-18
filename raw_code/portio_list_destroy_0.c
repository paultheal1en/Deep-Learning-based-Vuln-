void portio_list_destroy(PortioList *piolist)

{

    g_free(piolist->regions);

    g_free(piolist->aliases);

}
