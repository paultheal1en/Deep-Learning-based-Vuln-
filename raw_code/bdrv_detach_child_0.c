static void bdrv_detach_child(BdrvChild *child)

{

    if (child->next.le_prev) {

        QLIST_REMOVE(child, next);

        child->next.le_prev = NULL;

    }



    bdrv_replace_child(child, NULL, false);



    g_free(child->name);

    g_free(child);

}
